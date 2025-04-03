import random
from dataclasses import dataclass

import chess
import chess.engine
import chess.pgn
import numpy as np
import pyspiel
import torch
from chessenv import CChessEnv
from chessenv.sfa import SFArray
from open_spiel.python.rl_environment import Environment, StepType


@dataclass
class ChessAgent:
    def __call__(self, board: chess.Board) -> chess.Move: ...

    @property
    def name(self):
        return self.__class__.__name__


@dataclass
class StockfishAgent(ChessAgent):
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

    @property
    def name(self):
        return f"{self.__class__.__name__}({self.depth})"


@dataclass
class RandomAgent(ChessAgent):
    def __call__(self, board: chess.Board, max_sel: bool = False) -> chess.Move:
        random_moves = list(board.legal_moves)
        return np.random.choice(random_moves)


def chess_rollout(white_engine, black_engine):
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = white_engine(board)
        elif board.turn == chess.BLACK:
            move = black_engine(board)

        board.push(move)

    game = chess.pgn.Game().from_board(board)
    game.headers["White"] = white_engine.name
    game.headers["Black"] = black_engine.name
    return game


class CChessEnvTorch(CChessEnv):
    def reset(self):
        states, mask = super().reset()
        return torch.LongTensor(states), torch.FloatTensor(mask)

    def step(self, actions):
        actions = np.int32(actions.cpu().numpy())

        # TODO: Fix this
        states, mask, rewards, done = super().step(actions)

        return (
            torch.LongTensor(states),
            torch.FloatTensor(mask),
            torch.FloatTensor(rewards),
            torch.FloatTensor(done),
        )


class CChessEnvTorchAgainstSF(CChessEnvTorch):

    sfa = SFArray(1)

    def sample_opponent(self):
        state = self.get_state()
        tmasks = self.get_mask()
        moves = self.sfa.get_move_ints(state)

        for i in range(moves.shape[0]):
            m = moves[i]
            t = tmasks[i]

            m = np.clip(m, 0, len(t))

            if t[m] != 1 or random.random() < 0.4:
                probs = t / np.sum(t)
                moves[i] = np.random.choice(len(probs), p=probs)

        return moves


class CChessEnvTorchTwoPlayer(CChessEnv):
    def reset(self):
        states, mask = super().reset()
        return torch.LongTensor(states), torch.FloatTensor(mask)

    def step(self, actions):
        actions = np.int32(actions.cpu().numpy())
        done, rewards = self.push_moves(actions)

        rewards[(self.t >= self.max_step)] = 0

        done = torch.BoolTensor(done) | torch.BoolTensor(self.t >= self.max_step)

        self.reset_boards(np.int32(done.numpy()))

        states = self.get_state()
        mask = self.get_mask()
        return (
            torch.LongTensor(states),
            torch.FloatTensor(mask),
            torch.FloatTensor(rewards),
            done,
        )


class TTTEnv:
    def __init__(self, n):
        self.n = n
        self._envs = [Environment(pyspiel.load_game("tic_tac_toe")) for _ in range(n)]

    def reset(self):
        out = [e.reset() for e in self._envs]
        states = [o.observations["info_state"][0] for o in out]

        legal_mask = np.zeros((self.n, 9))
        for i, la in enumerate([o.observations["legal_actions"] for o in out]):
            for a in la:
                legal_mask[i, a] = 1.0

        return torch.LongTensor(np.stack(states)), torch.FloatTensor(legal_mask)

    def get_response(self, out, current_player):
        response = random.sample(out.observations["legal_actions"][current_player], 1)
        return np.array(response)

    def reset_board(self, env):
        out = env.reset()

        if random.random() < 0.5:
            new_player = env.get_state.current_player()
            response = random.sample(out.observations["legal_actions"][new_player], 1)
            out = env.step(np.array(response))
        return out

    def step(self, action):
        states = []
        legal_mask = np.zeros((self.n, 9))
        rewards = np.zeros((self.n,))
        dones = np.zeros((self.n,))

        for i, a in enumerate(action):
            a = np.array([a.item()])

            current_player = self._envs[i].get_state.current_player()

            step_out = self._envs[i].step(a)

            if step_out.step_type == StepType.LAST:
                dones[i] = 1.0
                rewards[i] = step_out.rewards[current_player]

                out = self.reset_board(self._envs[i])

            else:

                new_player = self._envs[i].get_state.current_player()
                response = self.get_response(step_out, new_player)

                out = self._envs[i].step(np.array(response))

                if out.step_type == StepType.LAST:
                    dones[i] = 1.0
                    rewards[i] = out.rewards[current_player]
                    out = self.reset_board(self._envs[i])

            current_player = self._envs[i].get_state.current_player()

            state = out.observations["info_state"][current_player]

            if current_player == 1:
                state = state[0:9] + state[18:] + state[9:18]

            states.append(state)

            for a in out.observations["legal_actions"]:
                legal_mask[i, a] = 1.0

        return (
            torch.LongTensor(np.stack(states)),
            torch.FloatTensor(legal_mask),
            torch.FloatTensor(rewards),
            torch.BoolTensor(dones),
        )


class TTTEnvTwoPlayer(TTTEnv):
    def step(self, action):
        states = []
        legal_mask = np.zeros((self.n, 9))
        rewards = np.zeros((self.n,))
        dones = np.zeros((self.n,))

        for i, a in enumerate(action):
            a = np.array([a.item()])

            current_player = self._envs[i].get_state.current_player()
            out = self._envs[i].step(a)

            if out.step_type == StepType.LAST:
                dones[i] = 1.0
                rewards[i] = out.rewards[current_player]
                out = self.reset_board(self._envs[i])

            current_player = self._envs[i].get_state.current_player()

            state = out.observations["info_state"][current_player]

            if current_player == 1:
                state = state[0:9] + state[18:] + state[9:18]

            states.append(state)

            for a in out.observations["legal_actions"]:
                legal_mask[i, a] = 1.0

        return (
            torch.LongTensor(np.stack(states)),
            torch.FloatTensor(legal_mask),
            torch.FloatTensor(rewards),
            torch.BoolTensor(dones),
        )


class ChessSpielEnv:
    def __init__(self, n):
        self.n = n
        self._envs = [Environment(pyspiel.load_game("chess")) for _ in range(n)]

    def reset(self):
        out = [e.reset() for e in self._envs]
        states = [o.observations["info_state"][0] for o in out]

        legal_mask = np.zeros((self.n, 9))
        for i, la in enumerate([o.observations["legal_actions"] for o in out]):
            for a in la:
                legal_mask[i, a] = 1.0

        return torch.LongTensor(np.stack(states)), torch.FloatTensor(legal_mask)

    def step(self, action):
        states = []
        legal_mask = np.zeros((self.n, 9))
        rewards = np.zeros((self.n,))
        dones = np.zeros((self.n,))
        for i, a in enumerate(action):
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
            torch.FloatTensor(rewards),
            torch.BoolTensor(dones),
        )
