from transformers import AutoModelForCausalLM, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from neuronx_distributed.pipeline.model import NxDPPModel


def get_traced_model_gpt():
    seq_len = 512
    model_config = GPT2Config(
        vocab_size=50257,
        n_positions=seq_len,
        n_embd=768,
        n_layer=8,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-05,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.0,
        use_cache=False,
        bos_token_id=50256,
        eos_token_id=50256,
        return_dict=False,
    )
    module = AutoModelForCausalLM.from_config(model_config)
    model = NxDPPModel(module=module, transformer_layer_cls=GPT2Block, tracer_cls="hf")
    model.trace(input_names=["input_ids", "attention_mask", "labels"], leaf_modules=['GPT2Block'])
    cut_points = ["transformer.h.1", "transformer.h.2", "transformer.h.3", "transformer.h.4", "transformer.h.5",
                  "transformer.h.6", "transformer.h.7"]
    for cut in cut_points:
        model.cut_pipeline_stage(cut)
    return model.traced_model
