from collections import deque
from dataclasses import dataclass
from typing import Any

import chess
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from stonefish.rep import MoveToken, MoveRep, BoardRep, CBoardRep
from chessplotlib import plot_board, plot_move, mark_move
from mllg import TestInfo, ValidationInfo

from stonefish.env import (
    CChessEnvTorch,
    TTTEnvTwoPlayer,
    CChessEnvTorchTwoPlayer,
    Stockfish,
)
from stonefish.utils import ttt_state_to_str
from chessenv.rep import CMove


@dataclass
class EvalContext:
    # Should be Two player
    eval_env: Any

    def __call__(self, model, batch_idx):
        win_per = eval_against_random(model, self.eval_env, N=100)
        return ValidationInfo(0, batch_idx, [TestInfo("WinPer", win_per)])


@dataclass
class TTTEvalContext(EvalContext):
    # Should be Two player
    eval_env: Any = TTTEnvTwoPlayer(1)

    def __call__(self, model, batch_idx):
        sample_games = ttt_walkthrough(model, self.eval_env, N=2)
        win_info = eval_against_random(model, self.eval_env, N=100)
        return ValidationInfo(0, batch_idx, [win_info, sample_games])


@dataclass
class ChessEvalContext:
    eval_env: Any = CChessEnvTorchTwoPlayer(1)

    def __call__(self, model, batch_idx):

        random_pgn = pgns_against_random_chess(model, self.eval_env, N=10)

        stock_pgn = pgns_against_stockfish_chess(model, self.eval_env, max_sel=True, N=1)
        stock_pgn = pgns_against_stockfish_chess(model, self.eval_env, max_sel=False, N=10)

        win_info = eval_against_random(model, self.eval_env, N=100)

        return ValidationInfo(0, batch_idx, [win_info, random_pgn, stock_pgn])


def print_example(model, states, actions, infer):
    for (s, a, i) in list(zip(states, actions, infer))[:16]:
        example = s[s != -1]
        board_str = model.input_rep.from_tensor(example).to_str()
        pred_str = model.output_rep.from_tensor(i).to_str()
        label_str = model.output_rep.from_tensor(a).to_str()
        print(f"{board_str} {pred_str} {label_str}")


def eval_model(model, datal, train_fn, max_batch=20):

    correct = 0.0
    total = 0.0
    losses = []

    for (batch_idx, (s, a)) in enumerate(datal):
        model.eval()

        infer = model.inference(s, a.shape[1] - 1)

        for i in range(len(infer)):
            pred_str = model.output_rep.from_tensor(infer[i]).to_str()
            label_str = model.output_rep.from_tensor(a[i]).to_str()

            total += 1.0
            if pred_str == label_str:
                correct += 1.0

        with torch.no_grad():
            loss = train_fn(model, s, a)

        losses.append(loss.item())

        if batch_idx == max_batch:
            break

    print_example(model, s, a, infer)

    acc = correct / total
    m_loss = np.mean(losses)
    return [TestInfo("ACC", acc), TestInfo("loss", m_loss)]


def seq_eval_model(model, datal, train_fn, max_batch=20):

    correct = 0.0
    total = 0.0
    losses = []

    for (batch_idx, (s, a)) in enumerate(datal):
        model.eval()

        infer = model.inference(s, a.shape[1] - 1)
        flat_infer = torch.flatten(infer[:, 1:])

        labels = torch.flatten(a[:, 1:]).to(infer.device)

        flat_infer = flat_infer[labels != -1]
        labels = labels[labels != -1]

        correct += torch.sum((flat_infer == labels).float())
        total += flat_infer.shape[0]

        with torch.no_grad():
            loss = train_fn(model, s, a)

        losses.append(loss.item())

        if batch_idx == max_batch:
            break

    print_example(model, s, a, infer)

    acc = correct / total
    m_loss = np.mean(losses)
    return [TestInfo("ACC", acc.item()), TestInfo("loss", m_loss)]


def random_action(masks):
    masks = masks.numpy()
    probs = masks / np.sum(masks, axis=1).reshape(-1, 1)
    actions = np.zeros(len(masks))
    for (i, p) in enumerate(probs):
        actions[i] = np.random.choice(len(p), p=p)
    return torch.LongTensor(actions)


def eval_against_random(model, env, N=100, max_sel=True):
    wins = 0
    for _ in range(N):

        state, legal_mask = env.reset()

        # Player 0 goes first, this will be immediately flipped in the loop
        player_id = 1

        done = [False]
        while not done[0]:
            player_id = (player_id + 1) % 2

            if player_id == 0:
                action, _ = model.sample(state, legal_mask, max_sel=max_sel)
            elif player_id == 1:
                action = random_action(legal_mask)

            state, legal_mask, reward, done = env.step(action)

        # Not sure how best to check for win conditions, but this seems about right?
        if (reward[0] == 1.0 and player_id == 0) or (
            reward[0] == -1.0 and player_id == 1
        ):
            wins += 1.0

    return TestInfo("Win Rate Against Random", wins / N)


def pgns_against_random_chess(model, env, N=10, max_sel=True):
    pgn_str = ""
    for _ in range(N):

        board = chess.Board()

        state, legal_mask = env.reset()

        # Player 0 goes first, this will be immediately flipped in the loop
        player_id = 1

        done = [False]
        while not done[0]:
            player_id = (player_id + 1) % 2

            if player_id == 0:
                action, _ = model.sample(state, legal_mask, max_sel=max_sel)
            elif player_id == 1:
                action = random_action(legal_mask)

            move = CMove.from_int(action[0].cpu().numpy()).to_move()
            board.push(move)
            state, legal_mask, reward, done = env.step(action)

        game = chess.pgn.Game.from_board(board)
        pgn_str += str(game) + "\n"

    return TestInfo("Game Against Random", pgn_str)


def pgns_against_stockfish_chess(model, env, N=10, max_sel=True, level=1):

    stockfish = Stockfish(level)

    pgn_str = ""

    for _ in range(N):

        board = chess.Board()

        state, legal_mask = env.reset()

        # Player 0 goes first, this will be immediately flipped in the loop
        player_id = 1

        done = [False]
        while not done[0]:
            player_id = (player_id + 1) % 2

            if player_id == 0:
                action, _ = model.sample(state, legal_mask, max_sel=max_sel)
            elif player_id == 1:
                board_rep = CBoardRep.from_tensor(state[0]).to_board()
                action = stockfish(board_rep)
                action = np.array([CMove.from_str(str(action)).to_int()])
                action = torch.LongTensor(action)

            move = CMove.from_int(action[0].cpu().numpy()).to_move()
            board.push(move)
            state, legal_mask, reward, done = env.step(action)

        game = chess.pgn.Game.from_board(board)
        pgn_str += str(game) + "\n"

    del stockfish

    return TestInfo("Game Against Stockfish", pgn_str)


def ttt_walkthrough(model, env, N=10, max_sel=True):
    sample_games = "\n"
    for _ in range(N):
        state, legal_mask = env.reset()

        # Player 0 goes first, this will be immediately flipped in the loop
        player_id = 1

        done = [False]
        while not done[0]:
            player_id = (player_id + 1) % 2

            if player_id == 0:
                action, _ = model.sample(state, legal_mask, max_sel=max_sel)
            elif player_id == 1:
                action = random_action(legal_mask)

            action = action.cpu().numpy()
            sample_games += ttt_state_to_str(state, action)

            state, legal_mask, reward, done = env.step(action)

        return TestInfo("Sample TTT Games", sample_games)
