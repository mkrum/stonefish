from dataclasses import dataclass
from collections import deque
from typing import List

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TrainEntry:
    epoch: int
    batch: int
    time: float
    loss: float

    @classmethod
    def from_line(cls, line, start_time=None):
        line_split = line.split()

        if line_split[0] != "TRAIN":
            raise ValueError(f"Not a train line: {line}")

        _, epoch, batch, time, loss = line_split

        if start_time is not None:
            time = float(time)
            time -= start_time

        return cls(int(epoch), int(batch), float(time), float(loss))


@dataclass(frozen=True)
class TestEntry:
    epoch: int
    time: float
    acc: float

    @classmethod
    def from_line(cls, line, start_time=None):
        line_split = line.split()

        if line_split[0] != "TEST":
            raise ValueError(f"Not a test line: {line}")

        _, epoch, time, acc = line_split

        if start_time is not None:
            time = float(time)
            time -= start_time

        return cls(int(epoch), float(time), float(acc))


def rolling_mean(lines: List, attr: str, window_size: int):

    window = deque(maxlen=window_size)
    means = []
    for line in lines:
        window.append(getattr(line, attr))
        means.append(np.mean(window))

    plt.plot(means)


def read_file(path):
    with open(path, "r") as progress_file:
        lines = progress_file.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def split_data(lines):
    train_lines = list(filter(lambda l: l[0:5] == "TRAIN", lines))
    test_lines = list(filter(lambda l: l[0:4] == "TEST", lines))
    return train_lines, test_lines


if __name__ == "__main__":
    path = "/nfs/fishtank/openai/basic.txt"

    lines = read_file(path)
    train_lines, test_lines = split_data(lines)

    start_time = TestEntry.from_line(test_lines[0]).time
    test_lines = [TestEntry.from_line(l, start_time=start_time) for l in test_lines]
    train_lines = [TrainEntry.from_line(l, start_time=start_time) for l in train_lines]
    for t in test_lines:
        print(t.acc)
    rolling_mean(train_lines, "loss", 100)
    plt.savefig("test.png")
