import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from stonefish.model import BaseModel
from stonefish.rep import BoardRep, MoveRep
from stonefish.dataset import ChessData
from stonefish.train import train_step


def test_overfit():

    device = torch.device("cpu")
    dataset = ChessData("test/sample.csv")
    dataloader = DataLoader(dataset, batch_size=3)

    model = BaseModel(
        device, BoardRep, MoveRep, num_encoder_layers=2, num_decoder_layers=2
    )

    opt = Adam(model.parameters(), lr=1e-4)
    state, action = next(iter(dataloader))

    loss = train_step(model, state, action)

    for _ in range(50):
        opt.zero_grad()
        loss = train_step(model, state, action)
        loss.backward()
        opt.step()

    assert loss.item() < 0.5
