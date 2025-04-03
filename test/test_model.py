import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from stonefish.dataset import ChessData
from stonefish.model import BaseModel
from stonefish.rep import BoardRep, MoveRep
from stonefish.train.base import seq_train_step


def test_overfit():

    device = torch.device("cpu")
    dataset = ChessData("test/sample.csv", BoardRep, MoveRep)
    dataloader = DataLoader(dataset, batch_size=3)

    model = BaseModel(
        device, BoardRep, MoveRep, num_encoder_layers=2, num_decoder_layers=2
    )

    opt = Adam(model.parameters(), lr=1e-4)
    state, action = next(iter(dataloader))

    for _ in range(50):
        opt.zero_grad()
        loss = seq_train_step(model, state, action)
        loss.backward()
        opt.step()

    assert loss.item() < 0.5
