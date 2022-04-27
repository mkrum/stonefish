import multiprocessing as mp
from dataclasses import dataclass
import random

import torch
import chess
import chess.engine
import chess.pgn
import numpy as np

import pyspiel
from open_spiel.python.rl_environment import Environment, StepType

from stonefish.model import BaseModel
from stonefish.rep import BoardRep, MoveRep


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

    def __del__(self):
        self.quit()


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
                import pdb

                pdb.set_trace()

            while move not in list(board.legal_moves):
                move = white_engine.sample(board)
        else:
            move = black_engine(board)

        board.push(move)
        moves += 1

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

        reward = 0.0

        if self.board.fullmove_number > 50 or self.board.halfmove_clock >= 99:
            return self.failure_reset()

        if not self.board.is_legal(move):
            reward -= 1.0
            move = np.random.choice(list(self.board.legal_moves))

        self.board.push(move)

        done = False
        if self.board.is_game_over():
            done = True
            reward += get_board_reward_white(self.board)
            self.board = chess.Board()
        else:
            response = self.eng(self.board)
            self.board.push(response)

            if self.board.is_game_over():
                done = True
                reward += get_board_reward_white(self.board)
                self.board = chess.Board()

        board_tensor = BoardRep.from_board(self.board).to_tensor().unsqueeze(0)
        return board_tensor, torch.FloatTensor([reward]), torch.BoolTensor([done])

    def reset(self):
        self.board = chess.Board()
        board_tensor = BoardRep.from_board(self.board).to_tensor().unsqueeze(0)
        return board_tensor


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

        procs = [
            mp.Process(target=worker, args=(self.state_qs[i], self.action_qs[i]))
            for i in range(self.n)
        ]

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


class TTTEnv:
    def __init__(self, n):
        self.n = n
        self._envs = [Environment(pyspiel.load_game("tic_tac_toe")) for _ in range(n)]

    def reset(self):
        out = [e.reset() for e in self._envs]
        states = [o.observations["info_state"][0] for o in out]

        legal_mask = np.zeros((self.n, 9))
        for (i, la) in enumerate([o.observations["legal_actions"] for o in out]):
            for a in la:
                legal_mask[i, a] = 1.0

        return torch.LongTensor(np.stack(states)), torch.FloatTensor(legal_mask)

    def step(self, action):
        states = []
        legal_mask = np.zeros((self.n, 9))
        rewards = np.zeros((self.n,))
        dones = np.zeros((self.n,))
        for (i, a) in enumerate(action):
            a = np.array([a.item()])

            out = self._envs[i].step(a)

            if out.step_type == StepType.LAST:
                dones[i] = 1.0
                rewards[i] = out.rewards[0]
                out = self._envs[i].reset()
            else:
                response = random.sample(out.observations["legal_actions"][1], 1)

                out = self._envs[i].step(np.array(response))

                if out.step_type == StepType.LAST:
                    dones[i] = 1.0
                    rewards[i] = out.rewards[0]
                    out = self._envs[i].reset()

            states.append(out.observations["info_state"][0])

            for a in out.observations["legal_actions"]:
                legal_mask[i, a] = 1.0

        return (
            torch.LongTensor(np.stack(states)),
            torch.FloatTensor(legal_mask),
            rewards,
            dones,
        )
