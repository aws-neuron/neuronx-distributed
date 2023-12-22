import os

import neuronperf as npf
import torch


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        input_ids, attention_mask = inputs
        # reset the model so it can generate multiple times
        self.model.reset()
        output_ids = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, 
            max_new_tokens=self.model.config.max_new_tokens, 
            top_k=1, do_sample=True
        )
        num_new_tokens = output_ids.numel() - input_ids.numel()
        return output_ids, num_new_tokens


def benchmark_sampling(batch_size, max_length, traced_model_path, tokenizer, model_load_fn):
    def load_fn(model_dir, **kwargs):
        model = model_load_fn(model_dir)
        generate_wrapper = ModelWrapper(model)
        return generate_wrapper

    def preprocess_fn(inputs):
        batch_encoding = tokenizer(
            inputs, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        input_ids = batch_encoding["input_ids"]
        attention_mask = batch_encoding["attention_mask"]
        return input_ids, attention_mask

    global num_total_new_tokens
    num_total_new_tokens = 0
    def postprocess_fn(outputs):
        output_ids, num_new_tokens = outputs
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        global num_total_new_tokens
        num_total_new_tokens += num_new_tokens
        return output

    def env_setup_fn(*_):
        del os.environ["NEURON_RT_VISIBLE_CORES"]

    prompt = ["I believe the meaning of life is"] * batch_size
    num_runs = 10
    reports = npf.benchmark(
        load_fn=load_fn,
        model_filename=traced_model_path,
        inputs=[prompt],
        batch_sizes=batch_size,
        n_models=1,
        max_infers=num_runs,
        max_duration=0,
        workers_per_model=1,
        env_setup_fn=env_setup_fn,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
        multiprocess=False,
    )

    report = reports[0]
    report["num_new_tokens_per_batch"] = round(num_total_new_tokens / num_runs / batch_size, 2)
    report["throughput_avg"] = round(report["num_new_tokens_per_batch"] * batch_size / (report["latency_ms_avg"] / 1000), 2)
    report["latency_per_token_ms_p50"] = round(report["latency_ms_p50"] / report["num_new_tokens_per_batch"], 2)
    report["latency_per_token_ms_p99"] = round(report["latency_ms_p99"] / report["num_new_tokens_per_batch"], 2)

    # display and save results
    npf.print_reports(reports, cols=["num_new_tokens_per_batch", "throughput_avg", "latency_per_token_ms_p50", "latency_per_token_ms_p99"])
    print(f"Results saved to: {npf.write_json(report)}")
    return report
