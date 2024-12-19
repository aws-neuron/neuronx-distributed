import copy
import json
import logging
import os
from neuronx_distributed.parallel_layers import utils
from transformers import AutoConfig, PretrainedConfig
from typing import Union
from .lora_serving import LoraServingConfig
CONFIG_FILE = "neuron_config.json"
LORA_CONFIG_FILE = "lora_config.json"


class NeuronConfig:
    """
    Base config class for inference in NxD.

    This class contains attributes that are needed for various inference
    optimization/features in NxD.
    """

    def __init__(self, hf_config: PretrainedConfig = None, lora_config: LoraServingConfig=None, **kwargs) -> None:
        self.hf_config: PretrainedConfig = hf_config

        # Basic config for inference in NxD
        self.batch_size = kwargs.pop("batch_size", 1)
        self.padding_side = kwargs.pop("padding_side", "right")
        # TODO: see if we can consolidate n_active_tokens and n_positions into one
        self.seq_len = kwargs.pop("seq_len", 128)
        self.n_active_tokens = kwargs.pop("n_active_tokens", self.seq_len)  # Need to provide example input shape for tracing
        self.n_positions = kwargs.pop("n_positions", self.seq_len)
        self.on_cpu = kwargs.pop("on_cpu", False)

        # fallback to sequence_length is for compatibility with vllm
        self.max_context_length = kwargs.pop("max_context_length", self.seq_len)
        self.max_new_tokens = kwargs.pop("max_new_tokens", self.seq_len - self.max_context_length)
        if self.max_new_tokens == 0:
            self.max_new_tokens = None
        self.max_length = kwargs.pop("max_length", self.seq_len)

        # Attention
        self.fused_qkv = kwargs.pop("fused_qkv", False)
        # TODO: Remove Llama attn_cls and multiple attention feature.
        self.attn_cls = "NeuronLlamaFDAttention" if kwargs.get('flash_decoding_enabled', False) \
            else "NeuronLlamaAttention"

        # Continuous batching
        # TODO: Check if we really need different batch size for CTE and TKG, given
        # that we anyway provide two different config instance for them.
        self.ctx_batch_size = kwargs.pop("ctx_batch_size", self.batch_size)
        self.tkg_batch_size = kwargs.pop("tkg_batch_size", self.batch_size)
        self.max_batch_size = kwargs.pop("max_batch_size", self.batch_size)
        self.is_continuous_batching = kwargs.pop("is_continuous_batching", False)

        # On-device sampling
        self.on_device_sampling = kwargs.pop("on_device_sampling", False)

        # Bucketing
        self.enable_bucketing = kwargs.pop("enable_bucketing", False)
        self.buckets = [self.seq_len]
        self.bucket_n_active_tokens = False

        # Quantization
        self.quantized = kwargs.pop("quantized", False)
        self.quantized_checkpoints_path = kwargs.pop("quantized_checkpoints_path", None)
        if self.quantized is True:
            assert self.quantized_checkpoints_path is not None, "quantized_checkpoints_path is required"
        self.quantization_type = kwargs.pop("quantization_type", "per_tensor_symmetric")
        self.quantization_dtype = kwargs.pop("quantization_dtype", None)
        # TODO: Add validation for quantized_checkpoints_path after the design discussions

        # Speculative decoding
        self.trace_tokengen_model = kwargs.pop("trace_tokengen_model", True)
        self.speculation_length = kwargs.pop("speculation_length", 0)
        self.spec_batch_size = self.batch_size

        # Medusa decoding
        self.is_medusa = kwargs.pop("is_medusa", False)
        self.medusa_speculation_length = kwargs.pop("medusa_speculation_length", 0)
        self.num_medusa_heads = kwargs.pop("num_medusa_heads", 0)
        self.medusa_tree = kwargs.pop("medusa_tree", 0)

        # Lora
        self.lora_config = lora_config

        # Distributed config
        self.tp_degree = kwargs.pop("tp_degree", 1)
        self.world_size = kwargs.pop("world_size", self.tp_degree)
        self.pp_degree = kwargs.pop("pp_degree", 1)
        self.ep_degree = kwargs.pop("ep_degree", 1)
        self.start_rank_id = kwargs.pop("start_rank_id", 0)
        self.local_ranks_size = kwargs.pop("local_ranks_size", self.world_size)

        # Flash decoding
        self.flash_decoding_enabled = kwargs.pop("flash_decoding_enabled", False)

        # this is num of neuron cores in each kv logical group. In flash decoding within one logical KV group (each
        # represents one unique KV head), the KVs are distributed along seq dim, in each core.
        if self.flash_decoding_enabled:
            # TODO: need work on padding for when below assertion fail
            assert hf_config.num_attention_heads % self.tp_degree == 0, \
                (f"expect num attention heads is multiple of tp degree but got {hf_config.num_attention_heads} "
                 f"and {self.tp_degree}")
            self.num_cores_per_group = utils.divide(min(self.tp_degree, hf_config.num_attention_heads),
                                                    hf_config.num_key_value_heads)

    def save(self, model_directory: Union[str, os.PathLike]):
        """
        Saves the config to a JSON file in the given model directory.
        """
        config_file = os.path.join(model_directory, CONFIG_FILE)
        self.to_json_file(config_file)
        if self.hf_config is not None:
            logging.debug(f"Saving HF config: {self.hf_config.to_json_string()}")
            self.hf_config.save_pretrained(model_directory)

        lora_config_file = os.path.join(model_directory, LORA_CONFIG_FILE)
        if self.lora_config is not None:
            logging.debug(f"Saving lora config: {self.lora_config.to_json_string()}")
            self.lora_config.to_json_file(lora_config_file)

    def to_json_file(self, json_file: Union[str, os.PathLike]):
        with open(json_file, "w", encoding="utf-8") as writer:
            config_json = self.to_json_string()
            logging.debug(f"Saving Neuron config: {config_json}")
            writer.write(config_json + "\n")

    def to_json_string(self) -> str:
        # HF config is serialized separately, so we exclude it.
        config_dict = copy.deepcopy(self.__dict__)
        del config_dict["hf_config"]
        del config_dict["lora_config"]
        return json.dumps(config_dict, indent=2, sort_keys=True)


    @classmethod
    def load(cls, model_directory: Union[str, os.PathLike], skip_hf_config=False, **kwargs) -> "NeuronConfig":
        """
        Loads the config from the given model directory.

        The given kwargs override any properties of the same name from the JSON file.

        This function uses AutoConfig to load the HuggingFace PretrainedConfig from the model directory.
        To load a PretrainedConfig for a custom config class, you must register your custom model with AutoConfig.
        * https://huggingface.co/docs/transformers/en/model_doc/auto#extending-the-auto-classes
        """
        config_file = os.path.join(model_directory, CONFIG_FILE)
        neuron_config = cls.from_json_file(config_file, **kwargs)
        if not skip_hf_config:
            neuron_config.hf_config = AutoConfig.from_pretrained(model_directory)
            logging.info(f"Loaded HF config: {neuron_config.hf_config.to_json_string()}")

        lora_config_file = os.path.join(model_directory, LORA_CONFIG_FILE)
        neuron_config.lora_config = LoraServingConfig.from_json_file(lora_config_file)

        return neuron_config

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike], **kwargs) -> "NeuronConfig":
        with open(json_file, "r", encoding="utf-8") as reader:
            neuron_config = cls.from_json_string(reader.read(), **kwargs)
            logging.info(f"Loaded Neuron config: {neuron_config.to_json_string()}")
            return neuron_config

    @classmethod
    def from_json_string(cls, json_string: str, **kwargs) -> "NeuronConfig":
        merged_kwargs = json.loads(json_string)
        merged_kwargs.update(kwargs)
        return cls(**merged_kwargs)


class MoENeuronConfig(NeuronConfig):
    """
    Base class for mixture of experts (MoE) config on Neuron.
    """
    def __init__(
            self,
            hf_config: PretrainedConfig = None,
            capacity_factor: float = None,
            glu_mlp: bool = True,
            **kwargs) -> None:
        self.capacity_factor = float(capacity_factor) if capacity_factor is not None else None
        self.glu_mlp = glu_mlp
        super().__init__(hf_config, **kwargs)
