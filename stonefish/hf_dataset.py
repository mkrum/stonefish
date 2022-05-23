
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

from stonefish.tokens import BoardTokenizer, MoveTokenizer, BoardMoveSeq2SeqTokenizer

train_dataset = load_dataset("csv", data_files="/nfs/one_train.csv")
test_dataset = load_dataset("csv", data_files="/nfs/one_test.csv")

tokenizer = BoardMoveSeq2SeqTokenizer()

def preprocess_function(examples):
    inputs = [example for example in examples['board']]
    targets = [example for example in examples['move']]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

tokenized_data = {}
tokenized_data['train'] = train_dataset.map(preprocess_function, batched=True)
tokenized_data['test'] = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train']['train'],
    eval_dataset=tokenized_data['test']['train'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
