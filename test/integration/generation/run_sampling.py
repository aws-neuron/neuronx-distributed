import argparse
import json
import torch_xla.debug.metrics as met

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
    StoppingCriteriaList,
    MaxLengthCriteria
)
import torch
import numpy as np


def main(args):
    if args.device == "xla":
        import torch_xla
        import torch_xla.core.xla_model as xm
        import neuronx_distributed
        device = xm.xla_device()
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    task_prompt = "translate English to German: "
    input_strings = [
        "How old are you?",
        "What is your name?",
        "We work at Amazon",
        "I am trying to test the generation method"
    ]

    results = []
    if device == "xla":
        xm.mark_step()
    for input_str in input_strings:
        encoder_input_ids = tokenizer(task_prompt+input_str, return_tensors="pt", padding="max_length", max_length=50).to(device)
        outputs = model.generate(
            **encoder_input_ids, max_new_tokens=20, use_cache=False
        )
        outputs = outputs.detach().cpu().numpy()
        if args.device == "cpu":
            outputs = np.pad(outputs, ((0,0),(0, 20-outputs.shape[-1])))
        results.append(outputs)
    
    with open(args.output_path, "wb") as f:
        np.save(f, np.array(results))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model",
        default="t5-small",
        help="Name of the model",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_path", default="out.json")
    parser.add_argument("--use_cache", default=False)
    args, _ = parser.parse_known_args()
    main(args)