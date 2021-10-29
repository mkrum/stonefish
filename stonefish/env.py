
import torch
import chess
import atexit
import pyspiel
import chess.engine
import chess.pgn

import numpy as np
from dataclasses import dataclass, field

from stonefish.model import Model
from stonefish.dataset import board_to_tensor, tensor_to_move

@dataclass
class Stockfish:
    """
    Agent wrapper for the stockfish engine
    """

    depth: int
    _engine: None = None
    
    def __post_init__(self):
        self._engine = chess.engine.SimpleEngine.popen_uci("stockfish")

    def __call__(self, board):
        result = self._engine.play(board, chess.engine.Limit(depth=self.depth))
        return result.move

    def quit(self):
        self._engine.quit()

@dataclass
class ModelEngine:
    """
    Agent wrapper for the stockfish engine
    """

    path: str
    _model: None = None
    
    def __post_init__(self):
        device = torch.device("cuda")
        model = Model(device, 128)
        model = model.to(model.device)
        model.load_state_dict(torch.load(self.path))
        self._model = model

    def __call__(self, board):
        tensor = board_to_tensor(board)
        tensor = tensor.unsqueeze(0)
        move = self._model.inference(tensor)
        move = tensor_to_move(move[0])
        return move

    def sample(self, board):
        tensor = board_to_tensor(board)
        tensor = tensor.unsqueeze(0)
        move = self._model.sample(tensor)
        move = tensor_to_move(move[0])
        return move

    def quit(self):
        self._engine.quit()

def run_game(white_engine, black_engine):
    board = chess.Board()
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = white_engine(board)
            while move not in list(board.legal_moves):
                move = white_engine.sample(board)

        else:
            move = black_engine(board)
        board.push(move)

    outcome = board.outcome()
    if outcome.winner or outcome.winner == None:
        print(chess.pgn.Game().from_board(board))
    if outcome.winner != None:
        return (int(outcome.winner), int( not outcome.winner))
    else:
        return (0.5, 0.5)

def get_board_reward_white(board):
    outcome = board.outcome()
    if outcome.winner:
        return int(outcome.winner)
    elif outcome.winner is None:
        return 0.0
    else:
        return -1

class _Env:

    def __init__(self):
        self.eng = Stockfish(1)
        self.reset()

    def step(self, move):

        move = tensor_to_move(move)
        if move not in list(self.board.legal_moves):
            reward = torch.FloatTensor([-1])
            done = torch.BoolTensor([True])
            self.board = chess.Board()
            board_tensor = board_to_tensor(self.board).unsqueeze(0)
            return board_tensor, done, reward

        self.board.push(move)

        reward = 0
        done = False
        if self.board.is_game_over():
            done = True
            reward = get_board_reward_white(self.board)
            self.board = chess.Board()
        else:
            response = self.eng(self.board)
            self.board.push(response)

            if self.board.is_game_over():
                done = True
                reward = get_board_reward_white(self.board)
                self.board = chess.Board()
        
        board_tensor = board_to_tensor(self.board).unsqueeze(0)
        return board_tensor, torch.BoolTensor([done]), torch.FloatTensor([reward])

    def reset(self):
        self.board = chess.Board()
        board_tensor = board_to_tensor(self.board).unsqueeze(0)
        return board_tensor

@dataclass(frozen=True)
class RolloutTensor:

    state: torch.FloatTensor
    action: torch.IntTensor
    reward: torch.FloatTensor
    done: torch.BoolTensor

    @classmethod
    def empty(cls):
        return cls(None, None, None, None)

    def __len__(self):
        return self.state.shape[0]

    def is_empty(self):
        return self.state == None

    def add(self, state, action, reward, done) -> "RolloutTensor":
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)
        reward = reward.unsqueeze(0)
        done = done.unsqueeze(0)

        if self.is_empty():
            return RolloutTensor(state, action, reward, done)

        new_state = torch.cat((self.state, state), 0)
        new_action = torch.cat((self.action, action), 0)
        new_reward = torch.cat((self.reward, reward), 0)
        new_done = torch.cat((self.done, done), 0)
        return RolloutTensor(new_state, new_action, new_reward, new_done)

    def stack(self, other) -> "RolloutTensor":

        if self.is_empty():
            return other
        
        new_state = torch.cat((self.state, other.state), 0)
        new_action = torch.cat((self.action, other.action), 0)
        new_reward = torch.cat((self.reward, other.reward), 0)
        new_done = torch.cat((self.done, other.done), 0)
        return RolloutTensor(new_state, new_action, new_reward, new_done)

    def decay_(self, gamma) -> "RolloutTensor":
        for i in reversed(range(len(self) - 1)):
            self.reward[i] = self.reward[i] + ~self.done[i] * gamma * self.reward[i + 1]

    def raw_rewards(self):
        return self.reward[self.done]

    def to(self, device):
        new_state = self.state.to(device)
        new_action = self.action.to(device)
        new_reward = self.reward.to(device)
        new_done = self.done.to(device)
        return RolloutTensor(new_state, new_action, new_reward, new_done)

#device = torch.device("cuda")
#model = Model(device, 128)
#model = model.to(model.device)
#model.load_state_dict(torch.load("size_12_128/model_23.pth"))
model = ModelEngine("unknown/model_19.pth")

import stonefish.test_utils as ut
eng = Stockfish(14)

total = 500

is_legal = 0.0
matches = 0.0

import tqdm
from stonefish.constants import *
for _ in tqdm.tqdm(range(total)):
    board = ut.randomize_board()
    eng_move = eng(board)
    move_str = np.random.choice(MOVE_TOKENS) + np.random.choice(MOVE_TOKENS)
    try:
        move = chess.Move.from_uci(move_str)
    except:
        continue
    model_moves = [move] #[model(board) for _ in range(1)]

    if eng_move in model_moves:
        matches += 1.0
    if any([m in list(board.legal_moves) for m in model_moves]):
        is_legal += 1.0

print(matches / total)
print(is_legal / total)

eng.quit()



#env = _Env()
#state = env.reset()
#done = False
#rt = RolloutTensor.empty()
#while not done:
#    action = model.sample(state)
#    print(action)
#    next_state, done, reward = env.step(action[0])
#    rt = rt.add(state, action, reward, done)
#    state = next_state
#
#rt.decay_(0.99)
#rt = rt.to(device)
#print(rt.state.unsqueeze(1).shape)
#print(rt.action.unsqueeze(1).shape)
#log_probs = model(rt.state.squeeze(1), rt.action.squeeze(1).long())
#print(log_probs.shape)
#log_probs = torch.gather(log_probs, 2, rt.action.squeeze(1).unsqueeze(-1).long())
#log_probs = torch.sum(log_probs, 1)
#print(log_probs.shape)
#loss = -1 * torch.mean(rt.reward * log_probs)
#print(loss)
