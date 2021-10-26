"""
Simple datase for the tokenized chess data
"""
import torch
import chess
from torch.utils.data import Dataset, DataLoader

from constants import MOVE_TOKENS, BOARD_TOKENS, BTOKEN_ID, MTOKEN_ID

def board_to_list(board):
    builder = []

    for square in chess.SQUARES_180:
        piece = board.piece_at(square)

        if piece:
            builder.append(piece.symbol())
        else:
            builder.append("e")
	
    if board.turn == chess.WHITE:
        builder.append("w")
    else:
        builder.append("b")

    return builder

def board_to_tensor(board):
    tokens = board_to_list(board)
    tensor = tokenize_board(tokens)
    return tensor

def tensor_to_move_str(move_tensor):
    return MOVE_TOKENS[int(move_tensor[0].item())] + MOVE_TOKENS[int(move_tensor[1].item())]

def tensor_to_move(move_tensor):
    str_value = tensor_to_move_str(move_tensor)
    try:
        move = chess.Move.from_uci(str_value)
    except ValueError:
        move = chess.Move.from_uci("0000")
    return move

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
