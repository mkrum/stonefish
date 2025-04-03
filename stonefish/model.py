"""
This contains the basic transformer model. It wraps the basic nn.Transformer
into a larger module that adds additional functionality.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional
from chessenv.rep import CMoves
from torch import Tensor
from torch.distributions.categorical import Categorical

from stonefish.mask import MoveMask
from stonefish.rep import MoveRep


def transformer_split_encode(
    self,
    src: Tensor,
    src_mask: Optional[Tensor] = None,
    src_key_padding_mask: Optional[Tensor] = None,
):

    memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    return memory


def transformer_split_decode(
    self,
    memory: Tensor,
    tgt: Tensor,
    tgt_mask: Optional[Tensor] = None,
    memory_mask: Optional[Tensor] = None,
    tgt_key_padding_mask: Optional[Tensor] = None,
    memory_key_padding_mask: Optional[Tensor] = None,
) -> Tensor:

    output = self.decoder(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    return output


def batched_index_select(input, dim, index):
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


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

    def _encode(self, src, src_key_padding_mask):
        """
        Single forward pass of the transformer.

        Passes the source tensor, src, and the target tensor, tgt, through the
        transformer to compute the output representation.
        """
        return transformer_split_encode(
            self.transformer, src, src_key_padding_mask=src_key_padding_mask
        )

    def _decode(self, memory, tgt, tgt_padding_mask):
        """
        Single forward pass of the transformer.

        Passes the source tensor, src, and the target tensor, tgt, through the
        transformer to compute the output representation.
        """
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(
            self.device
        )
        return transformer_split_decode(
            self.transformer,
            memory,
            tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

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

        logits = prelogits[:, :-1, :]

        if logit_mask is not None:
            logits = logits * logit_mask + (1 - logit_mask) * -1e8

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

        memory = self._encode(pos_embed_state, mask)

        for _ in range(max_len):
            decode = self._action_embed(tokens)
            tgt_mask = torch.zeros(decode.shape[0], decode.shape[1]).bool()
            tgt_mask = tgt_mask.to(self.device)

            out = self._decode(memory, decode, tgt_mask)

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


class EncoderOnly(nn.Module):
    def __init__(
        self,
        device,
        input_rep,
        output_rep,
        nlayers=6,
        nhead=4,
        d_model=256,
        d_hid=128,
    ):
        super().__init__()
        self.device = device
        self.input_rep = input_rep
        self.output_rep = output_rep

        self.pe = positionalencoding1d(d_model, 70).to(self.device)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, 0.0, batch_first=True
        )
        self.encode = transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers
        )

        self.embed = nn.Embedding(self.input_rep.width(), d_model)

        self.encode = transformer_encoder

        self.out = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_rep.width()),
        )

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, data, action):
        data = data.to(self.device)
        start = 25.0 * torch.ones(data.shape[0], 1).to(self.device)
        data = torch.cat([start.long(), data], dim=1)

        embeded_data = self.embed(data)
        pos_embed_data = embeded_data + self.pe[: embeded_data.shape[1]]

        rep = self.encode(pos_embed_data)
        return self.out(rep[:, 0])

    def inference(self, state, action):
        probs = self.forward(state, action)
        return torch.argmax(probs, dim=1)


class ClassModel(nn.Module):
    def __init__(self, device, input_rep, output_rep):
        super().__init__()
        self.device = device
        self.input_rep = input_rep
        self.output_rep = output_rep

        self.embed = nn.Embedding(30, 8).to(self.device)

        self.model = nn.Sequential(
            nn.Linear(8 * 69, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_rep.width()),
        ).to(self.device)

    def forward(self, state, action):
        state = state.to(self.device)
        embedded_state = self.embed(state)
        flat_es = embedded_state.view(
            -1, embedded_state.shape[1] * embedded_state.shape[2]
        )
        return self.model(flat_es)

    def inference(self, state, action):
        probs = self.forward(state, action)
        return torch.argmax(probs, dim=1)


class TBased(nn.Module):
    def __init__(self, device, input_rep, output_rep):
        super().__init__()

        self.device = device
        self.input_rep = input_rep
        self.output_rep = output_rep
        emb_dim = 64
        self.policy = BaseModel(
            device,
            input_rep,
            output_rep,
            emb_dim=emb_dim,
            num_decoder_layers=2,
            num_encoder_layers=4,
        ).to(self.device)

        self.policy = self.policy.to(self.device)

        self.V = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # self.policy.load_state_dict(torch.load("../chess_seq_final/model_2.pth"))
        self.load_state_dict(torch.load("../chess_rl_sf_5_cont/model_27500.pth"))

    def forward(self, state, action, logit_mask):
        action = torch.stack(
            [
                MoveRep.from_str(m).to_tensor()
                for m in CMoves.from_int(action.cpu().numpy()).to_str()
            ]
        ).to(self.device)

        full_mask = logit_mask.to(self.device)

        out, prelogits = self.policy(state, action, return_hidden=True)
        masked_prelogits = prelogits * full_mask + (1 - full_mask) * -1e8

        logits = functional.log_softmax(masked_prelogits, dim=-1)
        sel_logits = batched_index_select(logits, 2, action[:, 1:].unsqueeze(-1))
        total_logits = torch.sum(sel_logits, dim=1)

        values = self.V(out[:, 0].detach())

        return total_logits, values

    def value(self, state):
        action = torch.zeros((state.shape[0], 1)).long().to(self.device)
        out, _ = self.policy(state, action, return_hidden=True)
        values = self.V(out[:, 0].detach())
        return values

    @torch.no_grad()
    def sample(self, state, logit_mask, max_sel=False):
        logit_mask = logit_mask.to(self.device)

        saved_mask = torch.zeros(logit_mask.shape[0], 2, self.output_rep.width())
        move_mask = MoveMask.from_mask(logit_mask)

        state = state.to(self.device).long()

        state_mask = get_mask(state)
        pos_embed_state = self.policy._state_embed(~state_mask * state)

        start_token = self.policy.start_token.repeat(state.shape[0], 1)

        memory = self.policy._encode(pos_embed_state, state_mask)

        tokens = start_token.to(self.device)
        for i in range(2):
            decode = self.policy._action_embed(tokens)
            tgt_mask = torch.zeros(decode.shape[0], decode.shape[1]).bool()
            tgt_mask = tgt_mask.to(self.device)

            out = self.policy._decode(memory, decode, tgt_mask)

            logits = self.policy.to_dist(out)[:, -1, :]

            action_mask = move_mask.get_mask(tokens)
            saved_mask[:, i, :] = action_mask

            logits = action_mask * logits + (1 - action_mask) * -1e8
            logits = functional.log_softmax(logits, dim=1)

            if max_sel:
                next_value = torch.argmax(logits, dim=-1).view(-1, 1)
            else:
                next_value = Categorical(logits=logits).sample().view(-1, 1)

            tokens = torch.cat((tokens, next_value), dim=1)

            embed_next = self.policy.to_emb_move(self.policy.move_embed(next_value))
            decode = torch.cat((decode, embed_next), dim=1)

        moves = torch.LongTensor(
            [MoveRep.from_tensor(t).to_cmove().to_int() for t in tokens]
        )
        return moves, saved_mask


class ACBase(nn.Module):
    def __init__(
        self, device, input_rep, output_rep, load_policy=None, load_value=None
    ):
        super().__init__()
        self.device = device
        self.policy = ClassModel(
            device,
            input_rep,
            output_rep,
        ).to(self.device)

        self.policy.load_state_dict(
            torch.load("/nfs/bigclass/model_3.pth", map_location=self.device)
        )

        self.V = ClassModel(
            device,
            input_rep,
            output_rep,
        ).to(self.device)

        self.V.load_state_dict(
            torch.load("/nfs/bigclass/model_3.pth", map_location=self.device)
        )

        self.act = functional.log_softmax
        self.load_state_dict(torch.load("/nfs/class_rl_invert/model_600.pth"))

    def forward(self, state, action, logit_mask=None):
        state = state.to(self.device).long()
        logits = self.act(
            self.policy(state, action) * logit_mask + (1 - logit_mask) * -1e8,
            dim=-1,
        )
        values = self.V(state, None)[:, 0].view(-1, 1)
        logits = torch.gather(logits, 1, action)
        return logits, values

    def value(self, state):
        state = state.to(self.device)
        values = self.V(state, None)[:, 0].view(-1, 1)
        return values

    @torch.no_grad()
    def sample(self, state, move_mask, max_sel=False):
        state = state.to(self.device).long()
        move_mask = move_mask.to(self.device)
        logits = self.policy(state, None)
        logits = self.act(logits * move_mask + (1 - move_mask) * -1e8, dim=-1)
        if max_sel:
            actions = torch.argmax(logits, dim=1)
        else:
            actions = Categorical(logits=logits).sample()
        return actions, move_mask


class SimpleRL(nn.Module):
    def __init__(
        self,
        device,
    ):
        super().__init__()
        self.device = device
        self.policy = nn.Sequential(
            nn.Linear(27, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 9),
        )
        self.act = functional.log_softmax

        self.V = nn.Sequential(
            nn.Linear(27, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action, logit_mask=None):
        state = state.to(self.device).float()
        logits = self.act(
            self.policy(state) * logit_mask + (1 - logit_mask) * -1e8, dim=-1
        )
        values = self.V(state)
        logits = torch.gather(logits, 1, action)
        return logits, values

    def value(self, state):
        state = state.to(self.device).float()
        return self.V(state)

    @torch.no_grad()
    def sample(self, state, move_mask, max_sel=False):
        state = state.to(self.device).float()
        move_mask = move_mask.to(self.device)

        prelogits = self.policy(state)

        logits = self.act(prelogits * move_mask + (1 - move_mask) * -1e8, dim=-1)

        if max_sel:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = Categorical(logits=logits).sample()

        return actions, move_mask
