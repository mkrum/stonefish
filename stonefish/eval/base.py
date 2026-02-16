import numpy as np
import torch
from mllg import TestInfo


def print_example(model, states, actions, infer):

    # Handle DistributedDataParallel wrapper
    model_unwrapped = model.module if hasattr(model, "module") else model

    for idx in range(min(states.shape[0], 16)):
        s = states[idx]
        a = actions[idx]
        i = infer[idx]

        board_str = model_unwrapped.board_tokenizer.to_fen(s)
        pred_str = model_unwrapped.move_tokenizer.to_uci(i.cpu())
        label_str = model_unwrapped.move_tokenizer.to_uci(a.cpu())
        print(f"{board_str} {pred_str} {label_str}")


def eval_model(model, datal, train_fn, max_batch=20):

    correct = 0.0
    total = 0.0
    losses = []

    for batch_idx, (s, a) in enumerate(datal):
        model.eval()

        with torch.no_grad():
            # Move input to same device as model
            device = next(model.parameters()).device
            s = s.to(device)
            a = a.to(device)
            # Handle DistributedDataParallel wrapper
            model_unwrapped = model.module if hasattr(model, "module") else model
            infer = model_unwrapped.inference(s).argmax(dim=-1)

        correct += (infer == a).sum()
        # Assuming this is batch dim? Might need to change
        total += s.shape[0]

        with torch.no_grad():
            result = train_fn(model, s, a)
            # Handle both old (loss only) and new (loss, accuracy) return formats
            if isinstance(result, tuple):
                loss, _ = result
            else:
                loss = result

        losses.append(loss.item())

        if batch_idx == max_batch:
            break

    print_example(model, s, a, infer)

    acc = correct / total
    m_loss = np.mean(losses)
    return [TestInfo("ACC", float(acc)), TestInfo("loss", float(m_loss))]


def _create_pgn_html(pgn):
    header = """
<link rel="stylesheet" type="text/css" href="https://pgn.chessbase.com/CBReplay.css"/>
<script src="https://pgn.chessbase.com/jquery-3.0.0.min.js"></script>
<script src="https://pgn.chessbase.com/cbreplay.js" type="text/javascript"></script>

<div class="cbreplay">

"""
    return header + str(pgn) + "\n<div>"
