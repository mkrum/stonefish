"""
Base Transformer model
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical

from stonefish.rep import MoveToken, BoardToken, BoardRep, MoveRep

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class Model(nn.Module):
    """
    A transformer model that is as vanilla as the come. Using the exact
    defaults from the pytorch transfomer class.

    Using a learned positional encoding. Was originally going to try some
    clever 1d -> 2d mapping but since we have a fixed size input, thought it
    would be easier just to optimize the encoding as well.
    """

    def __init__(self, device, emb_dim, input_token, output_token):
        super().__init__()
        self.device = device

        self.board_embed = nn.Embedding(input_token.size(), 8)
        self.move_embed = nn.Embedding(output_token.size(), 8)

        self.pos_encoding = positionalencoding1d(emb_dim, 9).to(self.device)
        self.start_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        self.transformer = nn.Transformer(
            batch_first=True,
            d_model=emb_dim,
            num_encoder_layers=6,
            num_decoder_layers=6,
        )

        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.to_emb_board = nn.Sequential(
            nn.Linear(8, emb_dim)
        )

        self.to_emb_move = nn.Sequential(
            nn.Linear(8, emb_dim)
        )

        self.to_dist = nn.Sequential(
            nn.Linear(emb_dim, output_token.size())
        )

    def _state_embed(self, state):
        state = state.to(self.device)
        embed_state = self.to_emb_board(self.board_embed(state))
        pos_embed_state = embed_state + self.pos_encoding
        return pos_embed_state

    def _action_embed(self, action, start_token):
        action = action.to(self.device)
        tgt_embed = self.to_emb_move(self.move_embed(action))
        tgt_start_token = torch.cat((start_token, tgt_embed), dim=1)
        return tgt_start_token

    def _get_start_token(self, batch_size):
        repeat_token = self.start_token.repeat(batch_size, 1, 1)
        return repeat_token.to(self.device)

    def _transformer_pass(self, src, tgt):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(
            self.device
        )
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return out

    def forward(self, state, action):
        start_token = self._get_start_token(state.shape[0])

        pos_embed_state = self._state_embed(state)
        tgt_embed = self._action_embed(action, start_token)
        out = self._transformer_pass(pos_embed_state, tgt_embed[:, :2, :])
        logits = self.to_dist(out)
        return logits

    def _inference(self, state, action_sel):
        pos_embed_state = self._state_embed(state)

        decode = self._get_start_token(state.shape[0])
        tokens = torch.zeros((state.shape[0], 1)).to(self.device)

        act = nn.LogSoftmax(dim=-1)

        for i in range(2):
            out = self._transformer_pass(pos_embed_state, decode)
            logits = act(self.to_dist(out)[:, -1, :])

            next_value = action_sel(logits)

            tokens = torch.cat((tokens, next_value), dim=1)
            embed_next = self.to_emb_move(self.move_embed(next_value))
            decode = torch.cat((decode, embed_next), dim=1)

        return tokens[:, 1:]

    @torch.no_grad()
    def inference(self, state):
        def max_action_sel(logits):
            return torch.argmax(logits, dim=1).view(-1, 1)

        return self._inference(state, max_action_sel)

    @torch.no_grad()
    def sample(self, state):
        def sample_action_sel(logits):
            return Categorical(logits=logits).sample().view(-1, 1)

        return self._inference(state, sample_action_sel)
