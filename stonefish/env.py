import multiprocessing as mp
from collections import deque

import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import chess
import atexit
import pyspiel
import chess.engine
import chess.pgn

import numpy as np
from dataclasses import dataclass, field

from stonefish.model import BaseModel
from stonefish.rep import BoardRep, MoveRep
import stonefish.utils as ut



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
class RandomEngine:
    """
    Agent wrapper for the stockfish engine
    """

    def __call__(self, board):
        moves = list(board.legal_moves)
        return np.random.choice(moves)


@dataclass
class ModelEngine:
    """
    Agent wrapper for the stockfish engine
    """

    path: str
    device: str
    _model: None = None

    def __post_init__(self):
        device = torch.device(self.device)
        model = BaseModel(device, BoardRep, MoveRep, emb_dim=256)
        model = model.to(model.device)
        model.load_state_dict(torch.load(self.path, map_location=device))
        self._model = model

    def __call__(self, board):
        tensor = BoardRep.from_board(board).to_tensor()
        tensor = tensor.unsqueeze(0)
        move = self._model.inference(tensor, max_len=2)
        move = MoveRep.from_tensor(move[0]).to_uci()
        return move

    def sample(self, board):
        tensor = BoardRep.from_board(board).to_tensor()
        tensor = tensor.unsqueeze(0)
        move = self._model.sample(tensor, max_len=2)
        move = MoveRep.from_tensor(move[0]).to_uci()
        return move

    def quit(self):
        self._engine.quit()


def rollout(white_engine, black_engine):
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
        return (int(outcome.winner), int(not outcome.winner))
    else:
        return (0.5, 0.5)


def run_game(white_engine, black_engine):
    board = chess.Board()

    moves = 0
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            try:
                move = white_engine(board)
            except AssertionError:
                import pdb; pdb.set_trace()

            while move not in list(board.legal_moves):
                move = white_engine.sample(board)
        else:
            move = black_engine(board)

        board.push(move)
        moves +=1

        if board.halfmove_clock >= 100:
            break

    return chess.pgn.Game().from_board(board)

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
        self.eng = RandomEngine()
        self.reset()

    def failure_reset(self):
        reward = torch.FloatTensor([-1])
        done = torch.BoolTensor([True])
        self.board = chess.Board()
        board_tensor = BoardRep.from_board(self.board).to_tensor().unsqueeze(0)
        return board_tensor, reward, done

    def step(self, move):

        move = MoveRep.from_tensor(move).to_uci()

        if self.board.fullmove_number > 50 or self.board.halfmove_clock >= 99:
            return self.failure_reset()

        try:
            self.board.push(move)
        except AssertionError:
            return self.failure_reset()

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

        board_tensor = BoardRep.from_board(self.board).to_tensor().unsqueeze(0)
        return board_tensor, torch.FloatTensor([reward]), torch.BoolTensor([done])

    def reset(self):
        self.board = chess.Board()
        board_tensor = BoardRep.from_board(self.board).to_tensor().unsqueeze(0)
        return board_tensor


@dataclass(frozen=True)
class RolloutTensor:

    state: torch.FloatTensor
    action: torch.IntTensor
    next_state: torch.FloatTensor
    reward: torch.FloatTensor
    done: torch.BoolTensor

    @classmethod
    def empty(cls):
        return cls(None, None, None, None, None)

    def __len__(self):
        if self.state is None:
            return 0
        return self.state.shape[1]

    def is_empty(self):
        return self.state == None

    def add(self, state, action, next_state, reward, done) -> "RolloutTensor":

        if self.is_empty():
            return RolloutTensor(state, action, next_state, reward, done)

        new_state = torch.cat((self.state, state), 1)
        new_action = torch.cat((self.action, action), 1)
        new_next_state = torch.cat((self.next_state, next_state), 1)
        new_reward = torch.cat((self.reward, reward), 1)
        new_done = torch.cat((self.done, done), 1)
        return RolloutTensor(new_state, new_action, new_next_state, new_reward, new_done)

    def stack(self, other) -> "RolloutTensor":

        if self.is_empty():
            return other

        new_state = torch.cat((self.state, other.state), 1)
        new_action = torch.cat((self.action, other.action), 1)
        new_next_state = torch.cat((self.next_state, other.next_state), 1)
        new_reward = torch.cat((self.reward, other.reward), 1)
        new_done = torch.cat((self.done, other.done), 1)
        return RolloutTensor(new_state, new_action, new_next_state, new_reward, new_done)

    def decay_(self, gamma, values) -> "RolloutTensor":

        self.reward[:, -1] += ~self.done[:, -1] * gamma * values

        for i in reversed(range(len(self) - 1)):
            self.reward[:, i] = self.reward[:, i] + ~self.done[:, i] * gamma * self.reward[:, i + 1]

    def raw_rewards(self):
        return self.reward[self.done]

    def to(self, device):
        new_state = self.state.to(device)
        new_action = self.action.to(device)
        new_next_state = self.next_state.to(device)
        new_reward = self.reward.to(device)
        new_done = self.done.to(device)
        return RolloutTensor(new_state, new_action, new_next_state, new_reward, new_done)

def worker(state_q, action_q):
    env = _Env()

    state = env.reset()
    state_q.put(state)

    while True:
        action = action_q.get()
        out = env.step(action)
        state_q.put(out)

class StackedEnv:

    def _reset(self):
        self.action_qs = [mp.Queue() for _ in range(self.n)]
        self.state_qs = [mp.Queue() for _ in range(self.n)]
        procs = [mp.Process(target=worker, args=(self.state_qs[i], self.action_qs[i])) for i in range(self.n)]

        for p in procs:
            p.start()
    
    def __init__(self, n):
        self.n = n
        self._reset()

    def reset(self):
        states = [s.get() for s in self.state_qs]
        return torch.stack(states)

    def step(self, actions):
        for i, aq in enumerate(self.action_qs):
            aq.put(actions[i])

        out = [s.get() for s in self.state_qs]
        states, rewards, dones = zip(*out)
        return torch.stack(states), torch.stack(rewards), torch.stack(dones)


def batched_index_select(input, dim, index):
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

if __name__ == "__main__":
    mp.set_start_method('spawn')

    device = torch.device("cuda")
    
    policy = BaseModel(device, BoardRep, MoveRep, emb_dim=256)
    policy.load_state_dict(torch.load("/nfs/fishtank/openai/model_1.pth", map_location=device))
    policy = policy.to(device)
    
    value_model = nn.Sequential(
        nn.Linear(256, 1)
    ).to(device)

    env = StackedEnv(20)
    
    opt = optim.Adam([{'params':list(value_model.parameters()), 'lr':1e-2},
                     {"params": list(policy.parameters()), 'lr':5e-5}]
                     )
    
    pl_hist = deque(maxlen=100)
    vl_hist = deque(maxlen=100)
    progress = deque(maxlen=100)

    state = env.reset()
    for it in range(1000):
        
        history = RolloutTensor.empty()

        for _ in range(32):
            action = policy.sample(state.squeeze(1), max_len=2)
            next_state, reward, done = env.step(action.cpu())
            action = action.unsqueeze(1)

            history = history.add(state, action, next_state, reward, done)

            state = next_state

        out, _ = policy(state.squeeze(1).to(device), action.squeeze(1), return_hidden=True)
        decay_values = value_model(out[:, -1, :])

        history = history.to(device)
        history.decay_(0.99, decay_values.view(-1,).detach())
    

        term_rewards = history.reward[history.done]
        for r in term_rewards:
            progress.append(r.item())

        flat_state = history.state.view(-1, history.state.shape[-1])
        flat_next_state = history.state.view(-1, history.next_state.shape[-1])
        flat_action = history.action.view(-1, history.action.shape[-1])
        flat_reward = history.reward.view(-1,)
         

        out, full_logits = policy(flat_state, flat_action, return_hidden=True)
        out = out[:, -1, :]
        values = value_model(out)

        logits = batched_index_select(full_logits, 2, flat_action[:, 1:].unsqueeze(-1))
        logits = logits.squeeze(-1)

        opt.zero_grad()

        value_loss = F.mse_loss(values, flat_reward.unsqueeze(-1))
        policy_loss = -1.0 * torch.mean(flat_reward.unsqueeze(-1).cuda() * logits)

        loss = value_loss + policy_loss
        loss.backward()
        opt.step()

        pl_hist.append(policy_loss.item())
        vl_hist.append(value_loss.item())

        print(f'{it}: PL: {np.mean(pl_hist)} VL: {np.mean(vl_hist)} R: {np.mean(progress)}', flush=True)

        if it % 100 == 0:
            torch.save(policy.state_dict(), "/nfs/fishtank/openai/random.pth")

