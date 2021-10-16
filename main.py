import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

_pieces = [
    "r",
    "n",
    "b",
    "q",
    "k",
    "p",
]

BOARD_TOKENS = ["e", "w", "b"] + _pieces + [p.upper() for p in _pieces]

_columns = ["a", "b", "c", "d", "e", "f", "g", "h"]
_rows = list(map(str, range(1, 9)))

_promotion_pieces = ["r", "n", "b", "q"]
_top_promotions = list(
    map(
        lambda x: "".join(x),
        list(itertools.product(_columns, ["1"], _promotion_pieces)),
    )
)
_bottom_promotions = list(
    map(
        lambda x: "".join(x),
        list(itertools.product(_columns, ["8"], _promotion_pieces)),
    )
)

SQUARES = list(map(lambda x: x[0] + x[1], list(itertools.product(_columns, _rows))))
MOVE_TOKENS = SQUARES + _top_promotions + _bottom_promotions

BTOKEN_ID = {b: i for (i, b) in enumerate(BOARD_TOKENS)}
MTOKEN_ID = {m: i for (i, m) in enumerate(MOVE_TOKENS)}


class TokenizedChess(Dataset):
    def __init__(self, path):
        super().__init__()

        with open(path, "r") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        raw = self.data[i].rstrip().split(",")
        board_state = raw[:65]
        board_tokens = torch.LongTensor(list(map(BTOKEN_ID.__getitem__, board_state)))
        action = torch.LongTensor(list(map(MTOKEN_ID.__getitem__, raw[65:])))
        return board_tokens, action


from torch.distributions.categorical import Categorical


class Model(nn.Module):
    def __init__(self, device, emb_dim):
        super().__init__()
        self.device = device

        self.board_embed = nn.Embedding(len(BOARD_TOKENS), emb_dim)
        self.move_embed = nn.Embedding(len(MOVE_TOKENS), emb_dim)

        self.pos_encoding = nn.Parameter(torch.zeros(65, emb_dim))
        self.start_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.transformer = nn.Transformer(
            d_model=emb_dim,
            num_encoder_layers=1,
            num_decoder_layers=1,
            batch_first=True,
        )

        self.to_dist = nn.Sequential(nn.Linear(emb_dim, len(MOVE_TOKENS)))

    def forward(self, state, action):
        embed_state = self.board_embed(state)
        pos_embed_state = embed_state + self.pos_encoding
        tgt_embed = self.move_embed(action)

        repeat_token = self.start_token.repeat(action.shape[0], 1, 1)
        tgt_embed_token = torch.cat((repeat_token, tgt_embed), dim=1)

        tgt_mask = self.transformer.generate_square_subsequent_mask(3)

        out = self.transformer(pos_embed_state, tgt_embed_token, tgt_mask=tgt_mask)
        out = out[:, :2]
        logits = self.to_dist(out)
        return logits

    def inference(self, state):
        embed_state = self.board_embed(state)
        pos_embed_state = embed_state + self.pos_encoding

        decode = self.start_token.repeat(state.shape[0], 1, 1)
        tokens = torch.zeros((state.shape[0], 1))

        for i in range(2):
            tgt_mask = self.transformer.generate_square_subsequent_mask(i + 1)
            out = self.transformer(pos_embed_state, decode, tgt_mask=tgt_mask)
            logits = self.to_dist(out)[:, -1, :]
            next_value = Categorical(logits=logits).sample().view(-1, 1)
            tokens = torch.cat((tokens, next_value), dim=1)
            embed_next = self.move_embed(next_value)
            decode = torch.cat((decode, embed_next), dim=1)

        return tokens[:, 1:]


data = TokenizedChess("test.txt")
test = DataLoader(data, batch_size=16, drop_last=True)
loss_fn = nn.CrossEntropyLoss()

model = Model(torch.device("cpu"), 16)

opt = optim.Adam(model.parameters(), lr=1e-2)

for s, a in test:
    for _ in range(10000):
        labels = torch.flatten(a)
        infer = model.inference(s)
        acc = torch.mean((infer.flatten() == labels).float())
        print(acc)

        opt.zero_grad()
        p = model(s, a)
        p = p.view(-1, len(MOVE_TOKENS))
        loss = loss_fn(p, labels)
        loss.backward()
        print(loss)
        opt.step()
        print()
