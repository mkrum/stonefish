from collections import deque
import numpy as np


class RLDisplay:
    def __init__(self):
        self.losses = {}
        self.outcomes = deque(maxlen=1000)

    def handle_trainstep(self, data):
        batch_idx = data["batch_idx"]

        for loss in data["losses"]:
            lt = loss["loss_type"]

            if lt == "wins":
                self.outcomes += [1.0] * int(loss["loss"])
            elif lt == "ties":
                self.outcomes += [0.0] * int(loss["loss"])
            elif lt == "losses":
                self.outcomes += [-1.0] * int(loss["loss"])
            else:
                current_val = self.losses.get(lt, deque(maxlen=100))
                current_val.append(loss["loss"])
                self.losses[lt] = current_val

        loss_str = f"({batch_idx:04}) "
        for k in self.losses.keys():
            loss_val = round(np.mean(self.losses[k]), 2)
            loss_str += f"{k}: {loss_val} "

        mean_reward = round(np.mean(self.outcomes), 3)
        loss_str += f"reward: {mean_reward} "
        print(loss_str)

    def handle_val(self, data):
        epoch = data["epoch"]
        batch_num = data["batch_idx"]

        val_str = f"Epoch: {epoch} Batch: {batch_num} "

        for loss_info in data["losses"]:
            loss_type = loss_info["loss_type"]
            loss_val = loss_info["loss"]
            val_str += f"{loss_type}: {loss_val} "

        print()
        print(val_str)

    def handle(self, data):
        if data["type"] == "trainstep_info":
            self.handle_trainstep(data)

        elif data["type"] == "val_info":
            self.handle_val(data)
