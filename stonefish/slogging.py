import time
import torch
from collections import deque


class Logger:

    output_dir = None
    log_file = None
    losses = None

    epoch_num = 0

    @classmethod
    def init(cls, output_dir=".", log_file="."):
        cls.output_dir = output_dir
        cls.log_file = open(f"{output_dir}/{log_file}", "w")

    @classmethod
    def save_checkpoint(cls, model, opt):
        torch.save(model.state_dict(), f"{cls.output_dir}/model_{cls.epoch_num}.pth")
        torch.save(opt.state_dict(), f"{cls.output_dir}/opt.pth")

    @classmethod
    def test_output(cls, acc, loss):
        print(f"({cls.epoch_num}) Acc: {round(acc, 2)} Test Loss: {loss}")
        cls.log_file.write(f"TEST {cls.epoch_num} {time.time()} {acc}\n")

    @classmethod
    def epoch(cls):
        cls.epoch_num += 1
        cls.losses = deque(maxlen=1000)

    @classmethod
    def loss(cls, batch_idx, loss):
        cls.losses.append(loss)

        if batch_idx > 0 and batch_idx % 100 == 0:
            print(f"({cls.epoch_num}/{batch_idx}) Loss (Avg): {np.mean(losses)}")

        cls.log_file.write(f"TRAIN {cls.epoch_num} {batch_idx} {time.time()} {loss}\n")
