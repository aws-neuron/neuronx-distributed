import os
import argparse
from itertools import chain

from datasets import load_dataset
from transformers import AutoTokenizer
from functools import wraps
from huggingface_hub.hf_api import DatasetInfo

def handle_missing_keys(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'")
            kwargs[missing_key] = None
            return func(*args, **kwargs)
    return wrapper

# Monkey patch the DatasetInfo.__init__ method
DatasetInfo.__init__ = handle_missing_keys(DatasetInfo.__init__)

parser = argparse.ArgumentParser()
parser.add_argument('--llama-version', type=int, default=3, help='LLaMA version (default: 3)')

args = parser.parse_args()
llama_version = args.llama_version

dataset_name = "wikicorpus"
dataset_config_name = "raw_en"

block_size = 4096
save_path = f"~/examples_datasets/wikicorpus_llama{llama_version}_tokenized_4k"
if llama_version == 3:
    block_size = 8192
    save_path = f"~/examples_datasets/wikicorpus_llama{llama_version}_tokenized_8k"

tokenizer_path = os.getcwd()

save_path = os.path.expanduser(save_path)
tokenizer_path = os.path.expanduser(tokenizer_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

raw_datasets = load_dataset(dataset_name, dataset_config_name, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])


tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

if block_size > tokenizer.model_max_length:
    print("block_size > tokenizer.model_max_length")
block_size = min(block_size, tokenizer.model_max_length)


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {block_size}",
)

train_dataset = lm_datasets["train"]
print(len(train_dataset))

train_dataset.save_to_disk(save_path)
