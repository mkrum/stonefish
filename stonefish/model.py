"""
This contains the basic transformer model. It wraps the basic nn.Transformer
into a larger module that adds additional functionality.
"""
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from transformers import AutoModel, GPT2LMHeadModel, T5ForConditionalGeneration


def get_mask(data, padding_value=-100):
    """
    Computes the mask for the data.

    Here, we assume that every item with a value of padding_value is a mask. It
    then returns a boolean vector of all of the instances of padding_value.
    """
    mask = data == padding_value
    return mask.to(data.device)


def positionalencoding1d(d_model, length):
    """
    This is from: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py

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
    A very vanilla transformer model.

    This BaseModel wraps the nn.transformer. This class acts as an
    autoregressive model over the target conditioned on the state. This means
    that if we pass in a source of N tokens and a target of M tokens, it will
    return M log-probabilities, corresponding to the probability of generating
    the t+1th token in the state given the 1:t+1 tokens.
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

        self.pe = positionalencoding1d(emb_dim, 1024).to(self.device)

        self.transformer = nn.Transformer(
            batch_first=True,
            d_model=emb_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=0.0,
        )

        self.to_emb_board = nn.Sequential(nn.Linear(8, emb_dim))

        self.to_emb_move = nn.Sequential(nn.Linear(8, emb_dim))

        self.to_dist = nn.Sequential(
            nn.Linear(emb_dim, output_rep.width()),
        )

        self.out_act = nn.LogSoftmax(dim=-1)

        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.start_token = torch.tensor([start_id]).view(1, 1)

    def _encode_position(self, data):
        """Adds a positional encoding to a tensor"""

        if self.pe.device != data.device:
            self.pe = self.pe.to(data.device)

        return data + self.pe[: data.shape[1]]

    def _state_embed(self, state):
        """
        Converts the raw state a dense representation.

        Converts the long integer tensor first into the embedding, then
        projects that into a larger dense vector, and then encodes the position.
        """
        state = state.to(self.device)
        embed_state = self.to_emb_board(self.board_embed(state))
        pos_embed_state = self._encode_position(embed_state)
        return pos_embed_state

    def _action_embed(self, action):
        """
        Converts the raw actions into a dense representation

        Converts the long integer tensor first into the embedding, then
        projects that into a larger dense vector, and then encodes the position.
        """
        action = action.to(self.device)
        tgt_embed = self.to_emb_move(self.move_embed(action))
        pos_tgt_embed = self._encode_position(tgt_embed)
        return pos_tgt_embed

    def _transformer_pass(self, src, tgt, src_padding_mask, tgt_padding_mask):
        """
        Single forward pass of the transformer.

        Passes the source tensor, src, and the target tensor, tgt, through the
        transformer to compute the output representation.
        """

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

    def forward(self, state, action, return_hidden=False, logit_mask=None):
        """
        Returns the *shifted* logits for generating the action.
        """
        state = state.to(self.device)
        action = action.to(self.device)

        mask = get_mask(state)
        tgt_mask = get_mask(action)

        pos_embed_state = self._state_embed(~mask * state)
        tgt_embed = self._action_embed(~tgt_mask * action)
        out = self._transformer_pass(pos_embed_state, tgt_embed, mask, tgt_mask)
        prelogits = self.to_dist(out)

        prelogits = prelogits[:, :-1, :]

        if logit_mask is not None:
            prelogits = prelogits * logit_mask + (1 - logit_mask) * -1e8

        logits = self.out_act(prelogits)

        if return_hidden:
            return out, logits

        return logits

    def forward_with_hidden(self, state, action):
        """
        Returns the *shifted* logits for generating the action.
        """
        state = state.to(self.device)
        action = action.to(self.device)

        mask = get_mask(state)
        tgt_mask = get_mask(action)

        pos_embed_state = self._state_embed(~mask * state)
        tgt_embed = self._action_embed(~tgt_mask * action)
        out = self._transformer_pass(pos_embed_state, tgt_embed, mask, tgt_mask)
        logits = self.to_dist(out)
        return out, logits[:, :-1, :]

    def _inference(self, state, max_len, action_sel, move_mask=None):
        """Underlying inference function"""

        state = state.to(self.device)
        mask = get_mask(state)

        pos_embed_state = self._state_embed(~mask * state)

        start_token = self.start_token.repeat(state.shape[0], 1)
        tokens = start_token.to(self.device)

        for i in range(max_len):
            decode = self._action_embed(tokens)
            tgt_mask = torch.zeros(decode.shape[0], decode.shape[1]).bool()
            tgt_mask = tgt_mask.to(self.device)

            out = self._transformer_pass(pos_embed_state, decode, mask, tgt_mask)

            logits = self.to_dist(out)[:, -1, :]

            if move_mask is not None:
                logits, move_mask = move_mask.mask(logits, tokens)

            next_value = action_sel(self.out_act(logits))

            tokens = torch.cat((tokens, next_value), dim=1)
            embed_next = self.to_emb_move(self.move_embed(next_value))
            decode = torch.cat((decode, embed_next), dim=1)

        if move_mask:
            return tokens, move_mask.masks

        return tokens

    @torch.no_grad()
    def inference(self, state, max_len=2):
        """Returns the most likely actions for the given states"""

        def max_action_sel(logits):
            return torch.argmax(logits, dim=1).view(-1, 1)

        return self._inference(state, max_len, max_action_sel)

    @torch.no_grad()
    def sample(self, state, max_len, move_mask=None):
        """Samples an action via the distribution"""

        def sample_action_sel(logits):
            return Categorical(logits=logits).sample().view(-1, 1)

        return self._inference(state, max_len, sample_action_sel, move_mask=move_mask)


class ClassModel(nn.Module):
    def __init__(self, device, input_rep, output_rep):
        super().__init__()
        self.device = device
        self.input_rep = input_rep
        self.output_rep = output_rep

        self.embed = nn.Embedding(30, 2).to(self.device)

        self.model = nn.Sequential(
            nn.Linear(2 * 69, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_rep.width()),
            nn.LogSoftmax(dim=1),
        ).to(self.device)

    def forward(self, state, action):
        state = state.to(self.device)
        embedded_state = self.embed(state)
        flat_es = embedded_state.view(-1, embedded_state.shape[1] * 2)
        return self.model(flat_es)

    def inference(self, state, action):
        probs = self.forward(state, action)
        return torch.argmax(probs, dim=1)


class ACBase(nn.Module):
    def __init__(
        self, device, input_rep, output_rep, load_policy=None, load_value=None
    ):
        super().__init__()
        self.device = device
        self.policy = BaseModel(
            device,
            input_rep,
            output_rep,
            emb_dim=32,
            num_encoder_layers=3,
            num_decoder_layers=3,
        ).to(self.device)
        self.value = nn.Linear(32, 1).to(self.device)

        if load_policy:
            self.policy.load_state_dict(
                torch.load(load_policy, map_location=self.device)
            )

        if load_value:
            self.policy.load_state_dict(
                torch.load(load_value, map_location=self.device)
            )

    def forward(self, state, action, logit_mask=None):
        out, logits = self.policy.forward(
            state, action, logit_mask=logit_mask, return_hidden=True
        )
        Q_values = self.value(out[:, -1, :])
        return logits, Q_values

    def Q_value(self, state, action):
        out, _ = self.policy.forward(state, action, return_hidden=True)
        return self.value(out[:, -1, :])

    @torch.no_grad()
    def sample(self, state, move_mask=None):
        return self.policy.sample(state, max_len=2, move_mask=move_mask)
