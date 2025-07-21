import asyncio
from dataclasses import dataclass
from typing import List

import chess
import chess.engine
import numpy as np


@dataclass
class CentipawnValues:
    board: chess.Board
    moves: List[chess.Move]
    values: np.array


class StockFishEvaluator:

    def __init__(self, engine, depth: int, max_concurrency: int):
        self.max_concurrency = max_concurrency
        self.depth = depth
        self.engine = engine

        self.semaphore = asyncio.Semaphore(self.max_concurrency)

    async def evaluate(self, board: chess.Board, side_to_move: str) -> float:

        async with self.semaphore:
            result = await self.engine.analyse(
                board, chess.engine.Limit(depth=self.depth)
            )

        value = getattr(result["score"], side_to_move)().score()
        return float(value)


async def parse_board(evaluator, board) -> CentipawnValues:

    side_to_move = "white" if board.turn else "black"

    legal_moves = list(board.legal_moves)

    values = np.zeros(len(legal_moves))
    for idx, move in enumerate(legal_moves):
        board_copy = board.copy()
        board_copy.push(move)
        value = await evaluator.evaluate(board, side_to_move)
        values[idx] = value

    return CentipawnValues(board, legal_moves, values)


async def main():
    transport, engine = await chess.engine.popen_uci("stockfish")
    evaluator = StockFishEvaluator(engine, 13, 20)
    board = chess.Board()
    out = await parse_board(evaluator, board)
    print(out)


asyncio.run(main())
