"""
Simple datase for the tokenized chess data
"""
import torch
from torch.utils.data import Dataset, DataLoader

from constants import MOVE_TOKENS, BOARD_TOKENS, BTOKEN_ID, MTOKEN_ID

def tokenize_board(board_state):
    board_tokens = torch.LongTensor(list(map(BTOKEN_ID.__getitem__, board_state)))
    return board_tokens

def tokenize_action(actions):
    action_tokens = torch.LongTensor(list(map(MTOKEN_ID.__getitem__, actions)))
    return action_tokens

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
        actions = raw[65:]
        board_tokens = tokenize_board(board_state)
        action = tokenize_action(actions)
        return board_tokens, action
