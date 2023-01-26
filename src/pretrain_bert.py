from datasets import concatenate_datasets, load_dataset, load_from_disk, DatasetDict
from tqdm import tqdm
from transformers import (
    BertTokenizerFast,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import multiprocessing
from itertools import chain

num_proc = multiprocessing.cpu_count()

num_records = 50
bookcorpus = load_dataset("bookcorpus", split=f"train[:{num_records}]")
wiki = load_dataset("wikipedia", "20220301.en", split=f"train[:{num_records}]")
wiki = wiki.remove_columns(
    [col for col in wiki.column_names if col != "text"]
)  # only keep the 'text' column

assert bookcorpus.features.type == wiki.features.type
raw_datasets = concatenate_datasets([bookcorpus, wiki])
raw_datasets = raw_datasets.train_test_split(test_size=0.2, shuffle=True)


# create a tokenizer from existing one to re-use special tokens
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")


def tokenize_texts(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        return_special_tokens_mask=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return tokenized_inputs


# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= tokenizer.model_max_length:
        total_length = (
            total_length // tokenizer.model_max_length
        ) * tokenizer.model_max_length
    # Split by chunks of max_len.
    result = {
        k: [
            t[i : i + tokenizer.model_max_length]
            for i in range(0, total_length, tokenizer.model_max_length)
        ]
        for k, t in concatenated_examples.items()
    }
    return result


# preprocess dataset
tokenized_datasets = raw_datasets.map(
    tokenize_texts, batched=True, remove_columns=["text"], num_proc=num_proc
)

# print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
# print("=================")


tokenized_datasets = tokenized_datasets.map(
    group_texts, batched=True, num_proc=num_proc
)
# print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))

# shuffle dataset
tokenized_datasets = tokenized_datasets.shuffle(seed=34)
print(
    f"the dataset contains in total {len(tokenized_datasets)*tokenizer.model_max_length} tokens"
)

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    num_hidden_layers=12,
    intermediate_size=768,
    num_attention_heads=12,
)
model = BertForMaskedLM(config)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./model/bert_pretrain",
    overwrite_output_dir=True,
    num_train_epochs=10,
    auto_find_batch_size=True,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()
trainer.save_model("./model/bert_pretrain")

print("finish")
