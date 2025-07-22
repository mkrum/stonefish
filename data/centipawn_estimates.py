import asyncio
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import chess
import chess.engine
import datasets
import numpy as np
import tqdm.asyncio


@dataclass
class CentipawnValues:
    board: str
    moves: List[str]
    values: List[float]

    def to_json(self):
        return {"board": self.board, "moves": self.moves, "values": self.values}


class StockFishEvaluator:

    def __init__(self, depth: int, max_concurrency: int):
        self.max_concurrency = max_concurrency
        self.depth = depth
        self.engines: List[Tuple] = []
        self.engine_queue: asyncio.Queue = asyncio.Queue()

    async def init_engines(self):
        for _ in range(self.max_concurrency):
            transport, engine = await chess.engine.popen_uci("stockfish")
            self.engines.append((transport, engine))
            await self.engine_queue.put(engine)

    async def evaluate(self, board: chess.Board, side_to_move: str) -> float:
        engine = await self.engine_queue.get()

        result = await engine.analyse(board, chess.engine.Limit(depth=self.depth))

        value = getattr(result["score"], side_to_move)().score()
        mate = getattr(result["score"], side_to_move)().mate()

        if mate is not None:
            await self.engine_queue.put(engine)
            # Positive mate means current side wins, negative means loses
            return float(mate * 1000000)

        await self.engine_queue.put(engine)
        return float(value)

    async def cleanup(self):
        for _, engine in self.engines:
            await engine.quit()


async def parse_board(evaluator, board_fen: str) -> Optional[CentipawnValues]:

    board = chess.Board(board_fen)
    if board.is_game_over():
        return None

    side_to_move = "white" if board.turn else "black"

    legal_moves = list(board.legal_moves)

    values = np.zeros(len(legal_moves))
    for idx, move in enumerate(legal_moves):
        board_copy = board.copy()
        board_copy.push(move)
        value = await evaluator.evaluate(board_copy, side_to_move)
        values[idx] = value

    return CentipawnValues(board_fen, [str(m) for m in legal_moves], values.tolist())


async def process_boards_concurrent(evaluator, boards, max_concurrent=50):
    semaphore = asyncio.Semaphore(max_concurrent)
    seen_boards = set()

    async def process_one(board):
        async with semaphore:
            return await parse_board(evaluator, board)

    tasks = []
    for board in boards:
        if board in seen_boards:
            continue
        seen_boards.add(board)

        task = asyncio.create_task(process_one(board))
        tasks.append(task)

        # Yield completed tasks periodically
        if len(tasks) >= max_concurrent:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                result = await task
                if result:
                    yield result
            tasks = list(pending)

    # Process remaining tasks
    for task in asyncio.as_completed(tasks):
        result = await task
        if result:
            yield result


async def main():
    evaluator = StockFishEvaluator(6, 10)
    await evaluator.init_engines()

    data = datasets.load_dataset("mkrum/ParsedChess")["train"]
    output_file = open("depth_6_values.jsonl", "w")

    with tqdm.tqdm(total=len(data["board"]), desc="Processing boards") as pbar:
        async for result in process_boards_concurrent(evaluator, data["board"]):
            output_file.write(json.dumps(result.to_json()) + "\n")
            pbar.update(1)

    await evaluator.cleanup()
    output_file.close()


asyncio.run(main())
