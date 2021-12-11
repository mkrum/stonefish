import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from stonefish.slogging import Logger
from rich import print
from datasets import load_dataset, load_metric


def create_dataset(model, data):
    
    states = []
    actions = []

    for (batch_idx, (encoding, targets)) in enumerate(data):
        input_ids, attention_mask = encoding.input_ids.to(
            device
        ), encoding.attention_mask.to(device)

        label_mask = targets.attention_mask.to(device).flatten()
        labels = targets.input_ids.to(device)
    
        with torch.no_grad():
            outputs = get_lm_input(model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        state = outputs
        action = labels

        action = action.flatten()
        state = state.view(-1, 512)

        action = action[label_mask != 0]
        state = state[label_mask != 0]

        states.append(state)
        actions.append(action)

    return torch.cat(states), torch.cat(actions)

def get_lm_input(
    self,
    input_ids=None,
    attention_mask=None,
    decoder_input_ids=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
    encoder_outputs=None,
    past_key_values=None,
    inputs_embeds=None,
    decoder_inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    r"""
    """
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
    if head_mask is not None and decoder_head_mask is None:
        if self.config.num_layers == self.config.num_decoder_layers:
            warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
            decoder_head_mask = head_mask

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    hidden_states = encoder_outputs[0]

    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)

    if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(labels)

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)
        hidden_states = hidden_states.to(self.decoder.first_device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.decoder.first_device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.encoder.first_device)
        self.lm_head = self.lm_head.to(self.encoder.first_device)
        sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)

    return sequence_output
        

if __name__ == "__main__":

    from transformers import T5TokenizerFast
    import torch.optim as opt
    from torch.utils.data import DataLoader

    from transformers import T5Tokenizer, T5ForConditionalGeneration

    Logger.init("/tmp", "test.txt", True, log_freq=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    model.load_state_dict(torch.load("base.pth"))

    dataset = load_dataset("common_gen")

    def collate_fn(batch):
        concepts = [" ".join(b["concepts"]) for b in batch]
        targets = [b["target"] for b in batch]
        concepts = tokenizer(concepts, padding=True, return_tensors="pt")
        targets = tokenizer(targets, padding=True, return_tensors="pt")
        return concepts, targets

    train_dl = DataLoader(
        dataset["train"],
        batch_size=256,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        dataset["validation"],
        batch_size=512,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    states, actions = create_dataset(model, train_dl)
    torch.save(states.cpu(), "train_states.pth")
    torch.save(actions.cpu(), "train_actions.pth")

    states, actions = create_dataset(model, test_dl)
    torch.save(states.cpu(), "test_states.pth")
    torch.save(actions.cpu(), "test_actions.pth")
