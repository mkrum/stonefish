import optax
import shutil
import copy
import os
import jax
from collections import deque
from datasets import load_dataset
from transformers.models.t5.modeling_flax_t5 import FlaxT5ForConditionalGenerationModule, FlaxT5Stack
from transformers import (
    FlaxT5ForConditionalGeneration,
    T5Config,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import DataLoader
from flax.training.common_utils import onehot, shard
from flax.training import train_state
from flax import jax_utils
from typing import Any
import jax.numpy as jnp

import numpy as np
import flax.linen as nn

from stonefish.tokens import BoardTokenizer, MoveTokenizer, BoardMoveSeq2SeqTokenizer
from stonefish.eval.base import ChessEvalContext
from stonefish.eval.wrapper import ModelEvalWrapper
from transformers import AutoConfig
import wandb

board_tokenizer = BoardTokenizer()
move_tokenizer = MoveTokenizer()

class NonsharedFlaxT5ForConditionalGenerationModule(FlaxT5ForConditionalGenerationModule):

    def setup(self):
        self.model_dim = self.config.d_model

        self.in_embed = nn.Embed(
            board_tokenizer.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor),
        )

        self.out_embed = nn.Embed(
            move_tokenizer.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor),
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FlaxT5Stack(encoder_config, self.in_embed, dtype=self.dtype)

        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = FlaxT5Stack(decoder_config, self.out_embed, dtype=self.dtype)

        self.lm_head = nn.Dense(
            move_tokenizer.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

class NonsharedFlaxT5ForConditionalGeneration(FlaxT5ForConditionalGeneration):
     module_class = NonsharedFlaxT5ForConditionalGenerationModule

def prepare_data(example):
    board = example["board"]
    move = example["move"]

    board_tokens = board_tokenizer(board)
    move_tokens = move_tokenizer(move)
    board_tokens["decoder_input_ids"] = move_tokens["input_ids"]
    return board_tokens


def train_step(state, batch, rng):
    rng, new_rng = jax.random.split(rng)

    def loss_fn(params):
        out = state.apply_fn(**batch, params=params, train=True, dropout_rng=rng)

        labels = batch["decoder_input_ids"][:, 1:]
        logits = out.logits[:, :-1]

        loss = optax.softmax_cross_entropy(
            logits, onehot(labels, logits.shape[-1])
        ).mean()

        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels)
        full_accuracy = jnp.sum(accuracy, axis=-1) == 2

        return loss, (accuracy.mean(), full_accuracy.mean())

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    out, grad = grad_fn(state.params)

    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
            {"train/loss": out[0], "train/accuracy": out[1][0], "train/FullAccuracy": out[1][1], "train/learning_rate": linear_decay_lr_schedule_fn(state.step)},
        axis_name="batch",
    )
    return new_state, metrics, new_rng


def make_print():
    keys = ["train/loss", "train/accuracy", "train/FullAccuracy"]
    hist = {k: deque(maxlen=100) for k in keys}

    def print_metrics(metrics):
        for k in keys:
            hist[k].append(float(metrics[k]))

        for k in keys:
            print(f"{k}: {round(np.mean(hist[k]), 2)}", end=" ")
        print()

    return print_metrics


if __name__ == "__main__":
    config = T5Config() #AutoConfig.from_pretrained("t5-base")
    config.tie_word_embeddings = False

    tokenizer = BoardMoveSeq2SeqTokenizer()

    model = NonsharedFlaxT5ForConditionalGeneration(config)

    ctx = ChessEvalContext()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, return_tensors="np"
    )

    OUTPUT_DIR = "./t5chess"


    num_train_steps = 500_000
    warmup_steps = 10_000

    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=1e-4, transition_steps=warmup_steps
    )

    decay_fn = optax.linear_schedule(
        init_value=1e-4,
        end_value=0,
        transition_steps=num_train_steps - warmup_steps,
    )

    linear_decay_lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
    )

    optimizer = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
    )

    wandb.init(project="stonefish")

    rng = jax.random.PRNGKey(1636)

    batch_size = 256 * jax.device_count()

    state = train_state.TrainState.create(
        apply_fn=model.__call__, params=model.params, tx=optimizer
    )

    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    state = jax_utils.replicate(state)
    rng = jax.random.split(rng, jax.device_count())

    # ?
    num_of_hosts = jax.process_count()
    current_host_idx = jax.process_index()

    pm = make_print()


    data_files = os.listdir("../data")
        
    step = 0
    for epoch in range(10):
        
        for data_file in data_files:

            dataset = load_dataset(
                "csv",
                data_files={"train": f'../data/{data_file}'},
                column_names=["board", "move"],
            )

            dataset = dataset.map(
                prepare_data,
                batched=False,
                num_proc=80,
                keep_in_memory=True,
                load_from_cache_file=False,
            )

            train_dl = DataLoader(
                dataset["train"],
                batch_size=batch_size,
                collate_fn=data_collator,
                drop_last=True,
                shuffle=True,
            )

            for (idx, batch) in enumerate(train_dl):
                step += 1
                del batch["board"]
                del batch["move"]
                del batch["token_type_ids"]

                local_host_model_inputs = {
                    key: np.split(batch.data[key], num_of_hosts, axis=0)[current_host_idx]
                    for key, value in batch.data.items()
                }

                model_inputs = shard(local_host_model_inputs)
                state, metrics, rng = p_train_step(state, model_inputs, rng)

                metrics = jax_utils.unreplicate(metrics)

                if step % 25 == 0:
                    metrics["train/step"] = step
                    metrics["train/epoch"] = epoch
                    wandb.log(metrics)

                pm(metrics)

                if step % 10000 == 0 and step > 0:
                    if jax.process_index() == 0:
                        params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                        ctx(ModelEvalWrapper(model, params), step)
                        model.save_pretrained(OUTPUT_DIR, params=params)
                        tokenizer.save_pretrained(OUTPUT_DIR)
            del dataset
            del train_dl
            shutil.rmtree("../.cache/huggingface/datasets")

        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
            model.save_pretrained(OUTPUT_DIR + f"_{epoch}", params=params)
            tokenizer.save_pretrained(OUTPUT_DIR + f"_{epoch}")

