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


def get_mask(data):
    mask = data == -1
    return mask.to(data.device)


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class BaseModel(nn.Module):
    """
    A transformer model that is as vanilla as the come. Using the exact
    defaults from the pytorch transfomer class.

    Using a learned positional encoding. Was originally going to try some
    clever 1d -> 2d mapping but since we have a fixed size input, thought it
    would be easier just to optimize the encoding as well.
    """

    def __init__(
        self,
        device,
        input_rep,
        output_rep,
        emb_dim=128,
        num_encoder_layers=6,
        num_decoder_layers=6,
        start_id=0,
    ):
        super().__init__()
        self.device = device

        self.input_rep = input_rep
        self.output_rep = output_rep

        self.board_embed = nn.Embedding(input_rep.width(), 8)
        self.move_embed = nn.Embedding(output_rep.width(), 8)

        self.pe = positionalencoding1d(emb_dim, 10).to(self.device)

        self.transformer = nn.Transformer(
            batch_first=True,
            d_model=emb_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=0.0,
        )

        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.to_emb_board = nn.Sequential(nn.Linear(8, emb_dim))

        self.to_emb_move = nn.Sequential(nn.Linear(8, emb_dim))

        self.to_dist = nn.Sequential(
            nn.Linear(emb_dim, output_rep.width()), nn.LogSoftmax(dim=-1)
        )

        self.start_token = torch.tensor([start_id]).view(1, 1)

    def _encode_position(self, data):

        if self.pe.device != data.device:
            self.pe = self.pe.to(data.device)

        return data + self.pe[: data.shape[1]]

    def _state_embed(self, state):
        state = state.to(self.device)
        embed_state = self.to_emb_board(self.board_embed(state))
        pos_embed_state = self._encode_position(embed_state)
        return pos_embed_state

    def _action_embed(self, action):
        action = action.to(self.device)
        tgt_embed = self.to_emb_move(self.move_embed(action))
        pos_tgt_embed = self._encode_position(tgt_embed)
        return pos_tgt_embed

    def _transformer_pass(self, src, tgt, src_padding_mask, tgt_padding_mask):

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(
            self.device
        )

        out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return out

    def forward(self, state, action):
        mask = get_mask(state)
        tgt_mask = get_mask(action)

        pos_embed_state = self._state_embed(state)
        tgt_embed = self._action_embed(action)
        out = self._transformer_pass(pos_embed_state, tgt_embed, mask, tgt_mask)
        logits = self.to_dist(out)
        return logits[:, :-1, :]

    def _inference(self, state, max_len, action_sel):
        mask = get_mask(state)

        pos_embed_state = self._state_embed(state)

        start_token = self.start_token.repeat(state.shape[0], 1)
        tokens = start_token

        for i in range(max_len):
            decode = self._action_embed(tokens)
            tgt_mask = torch.zeros(decode.shape[0], decode.shape[1]).bool()

            out = self._transformer_pass(pos_embed_state, decode, mask, tgt_mask)

            logits = self.to_dist(out)[:, -1, :]

            next_value = action_sel(logits)

            tokens = torch.cat((tokens, next_value), dim=1)
            embed_next = self.to_emb_move(self.move_embed(next_value))
            decode = torch.cat((decode, embed_next), dim=1)

        return tokens

    @torch.no_grad()
    def inference(self, state, max_len):
        def max_action_sel(logits):
            return torch.argmax(logits, dim=1).view(-1, 1)

        return self._inference(state, max_len, max_action_sel)

    @torch.no_grad()
    def sample(self, state, max_len):
        def sample_action_sel(logits):
            return Categorical(logits=logits).sample().view(-1, 1)

        return self._inference(state, max_len, sample_action_sel)
