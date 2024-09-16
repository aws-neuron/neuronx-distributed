import os
from os.path import join
import argparse
from itertools import chain

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='Name of the HF dataset to load.',
    )
    group.add_argument(
        '--json-keys', nargs='+', default=['text'], help='space separate listed of keys to extract from json'
    )
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument(
        '--tokenizer-model', type=str, default=None, help='Path to tokenizer model.',
    )
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output_path', type=str, required=True, help='Path to tokenized data set.')
    group.add_argument('--block_size', type=int, default=2048, help='Block size of input.')

    args = parser.parse_args()
    return args


def sample_num_spans(rng, max_num_spans=16, num_samples=1):
    # Choose at leat 1 span
    num_spans = rng.poisson(1.0, size=(num_samples)) + 1
    num_spans = [np.min([x, max_num_spans]) for x in num_spans]
    return num_spans


def format_to_infill(tokens, num_spans, tokenizer, rng):
    # Choose at leat 1 span
    # num_spans = rng.poisson(1.0) + 1
    # num_spans = np.min([num_spans, max_num_spans])

    # Based on the number of spans, a span can
    # have a size of maximum (len(text) - num_spans) // num_spans
    # This assumes we want at least one letter between two spans.
    max_len = (len(tokens) - num_spans) // num_spans

    # Handle cases of extremely short code.
    if max_len <= 0:
        return None

    # Set first start and end index based `max_len`
    max_start_idx = 1  # Very first letter will be skipped
    max_end_idx = max_len

    prefix_tokens = []
    suffix_tokens = []

    for span_idx in range(num_spans):
        # Randomly sample the length of the span
        sampled_length = rng.integers(1, max_len + 1)

        # Define low and high to sample a start position.
        # The first `+ 1` is due to `.integers` expecting a value
        # one higher than the one we want to sample. The second
        # `+ 1` is due to slicing in the next step.
        low, high = max_start_idx, max_end_idx - sampled_length + 1 + 1
        chosen_start = rng.integers(low, high)

        # Extract span
        span = tokens[chosen_start:chosen_start + sampled_length]

        # Add to prefix and suffix strings.
        # The masking begins with <mask_1>.
        prefix_tokens += tokens[max_start_idx - 1:chosen_start] + tokenizer.encode(f'<mask_{span_idx + 1}>')
        suffix_tokens += tokenizer.encode(f'<mask_{span_idx + 1}>') + span + tokenizer.encode('<eom>')

        # Update start and end indices
        max_start_idx = chosen_start + sampled_length + 1
        max_end_idx = chosen_start + sampled_length + max_len

        # Append leftover to prefix string
        if span_idx == num_spans - 1:
            prefix_tokens += tokens[chosen_start + sampled_length:]

    return prefix_tokens + tokenizer.encode('<|endoftext|><sep>') + suffix_tokens


def main(args):
    block_size = args.block_size
    tokenizer_path = os.path.expanduser(args.tokenizer_model)

    raw_datasets = load_dataset(args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    column_names = raw_datasets["train"].column_names
    code_column_name = 'content'

    def tokenize_function(examples):
        # 1. Encode the full batch
        encoded = tokenizer(examples[code_column_name])

        # 2. For each sample determine how many spans to mask
        num_samples = len(encoded['input_ids'])
        rng = np.random.default_rng(42)
        batch_num_spans = sample_num_spans(rng, num_samples=num_samples)

        infilled_batch = []
        attention_batch = []
        for i, (input_ids, attn_mask) in enumerate(zip(encoded['input_ids'], encoded['attention_mask'])):
            # Perform infilling only for 50% of the samples
            if rng.random() < 0.5:
                num_spans = batch_num_spans[i]
                # For each span, we introduce two <mask_X> tokens and one <eom> token.
                num_add_tokens = 3 * num_spans + 2

                # To ensure grouping the data does not break apart spans,
                # we only apply infilling on the first subset of tokens.
                tokens_for_infilling = input_ids[:block_size - num_add_tokens]

                # Perform infilling
                infilled_code = format_to_infill(tokens_for_infilling, num_spans, tokenizer, rng)
                if infilled_code is not None:
                    infilled_batch.append(infilled_code)
                    attention_batch.append([1] * len(infilled_code))
                else:
                    infilled_batch.append(input_ids)
                    attention_batch.append(attn_mask)
            else:
                infilled_batch.append(input_ids)
                attention_batch.append(attn_mask)

        return {'input_ids': infilled_batch, 'attention_mask': attention_batch}

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
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets['train']
    test_dataset = lm_datasets['test']
    valid_dataset = lm_datasets['valid']

    print(f"{len(train_dataset)=}")
    print(f"{len(test_dataset)=}")
    print(f"{len(valid_dataset)=}")

    train_save_path = os.path.expanduser(join(args.output_path, 'train'))
    test_save_path = os.path.expanduser(join(args.output_path, 'test'))
    valid_save_path = os.path.expanduser(join(args.output_path, 'valid'))

    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)
    os.makedirs(valid_save_path, exist_ok=True)

    train_dataset.save_to_disk(train_save_path)
    test_dataset.save_to_disk(test_save_path)
    valid_dataset.save_to_disk(valid_save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
