import json
import torch
from typing import List
from neuronx_distributed.parallel_layers.checkpointing import _invoke_preshard_hook
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel, 
    PretrainedConfig,
    set_seed,
    ViTImageProcessor,
    ViTModel
)

from runner import (
    InferenceRunner, 
    IMAGE_ENCODING_MODEL, 
    BENCHMARK_REPORT_FILENAME, 
    Benchmark, 
    generate_report
)
from vit.neuron_modeling_vit import (
    NeuronViTConfig,
    NeuronViTModel,
    NeuronViTForImageEncoding
)


class ViTRunner(InferenceRunner):
    
    def get_hf_config(self, sequence_length, **kwargs):
        hf_config: PretrainedConfig = AutoConfig.from_pretrained(self.model_path)
        hf_config.torch_dtype = torch.float32
        hf_config.vocab_size = 10000
        return hf_config

    def get_config_cls(self):
        return NeuronViTConfig

    def load_hf_model(self):
        return NeuronViTForImageEncoding.load_hf_model(self.model_path)

    def load_neuron_model_on_cpu(self, batch_size, **kwargs):
        self.hf_config = self.get_hf_config()
        self.neuron_config = self.get_config_for_nxd(
            self.hf_config,
            batch_size,
            1,
            max_prompt_length=0,         # not used in the ViT implementation
            sequence_length=0,            # not used in the ViT implementation
            enable_bucketing=False,         # not used in the ViT implementation
            **kwargs)
        
        self.hf_config.torch_dtype = torch.float32

        neuron_model = NeuronViTModel(self.neuron_config)
        state_dict = NeuronViTForImageEncoding.get_state_dict(self.model_path, neuron_config=self.neuron_config)
        _invoke_preshard_hook(neuron_model, state_dict)
        
        neuron_model.load_state_dict(state_dict, strict=False)
        
        if self.hf_config.torch_dtype == torch.bfloat16:
            neuron_model.bfloat16()

        model = NeuronViTForImageEncoding(None, self.neuron_config)
        model.image_encoding_model = neuron_model

        return model

    def load_neuron_model(self, traced_model_path):
        self.neuron_config = NeuronViTConfig.load(traced_model_path)
        model = NeuronViTForImageEncoding.from_pretrained(traced_model_path, self.neuron_config)
        
        model.load(traced_model_path)
        if self.neuron_config.hf_config.torch_dtype == torch.bfloat16:
            model.bfloat16()

        return model

    def load_tokenizer(self, padding_side=None):
        """
        Note used in the ViT implementation
        """
        return None
    
    def get_padding_side(self):
        """
        Note used in the ViT implementation
        """
        pass
    
    def get_default_hf_generation_config_kwargs(self):
        """
        Note used in the ViT implementation
        """
        pass
    
    def generate_on_neuron(self, prompts: List[str], model: PreTrainedModel, draft_model: PreTrainedModel = None, **kwargs):
        raise NotImplementedError("This function (generate_on_neuron) is disabled in ViTRunner.")
    
    def generate_on_cpu(self, prompts: List[str], batch_size: int, max_prompt_length: int, sequence_length: int, **kwargs):
        raise NotImplementedError("This function (generate_on_cpu) is disabled in ViTRunner.")
    
    def generate(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        prompts: List[str],
        max_length: int,
        draft_model: PreTrainedModel = None,
        **kwargs
    ):
        raise NotImplementedError("This function (generate) is disabled in ViTRunner.")
    
    def load_image_processor(self):
        return ViTImageProcessor.from_pretrained(self.model_path)
    
    def load_processor(self):
        return self.load_image_processor()

    def get_model_cls(self):
        return NeuronViTForImageEncoding
  
    def inference_with_hf_model(self, batch_size, **kwargs):
        """
        Use inference_with_hf_model to confirm the neuron model output 
        is exactly the same as hugging face model's ouput.
        """
        kwargs = self.fetch_pixel_values_for_kwargs(batch_size, **kwargs)
        model = ViTModel.from_pretrained(self.model_path)
        outputs = model(**kwargs)
        sequence_output = outputs[0]
        return sequence_output[:, 0, :]
    
    def inference_on_cpu(self, batch_size, **kwargs):
        """
        Use inference_on_cpu to confirm the neuron wrapper is correct. If the wrapper works
        on CPU, then the trace should work too. If it does not, it indicates a problem with
        the trace itself.
        """
        model = self.load_neuron_model_on_cpu(batch_size, **kwargs)
        outputs = self.inference(batch_size, model, **kwargs)
        model.reset()
        return outputs

    def inference_on_neuron(self, batch_size, model: PreTrainedModel, **kwargs):
        """
        Runs the trace on Neuron.
        """
        if not isinstance(model, PreTrainedModel):
            raise ValueError(f"Model should be of type PreTrainedModel, got type {type(model)}")

        with self.torch_profile(chrome_trace_path="generate-on-neuron.torch-trace.json"):
            outputs = self.inference(batch_size, model, **kwargs)
        model.reset()
        return outputs
    
    def fetch_pixel_values_for_kwargs(self, batch_size: int, **kwargs):
        """
        Add a key called pixel_values for the kwargs if the existing kwargs does not have it
        """
        # fetch and parse image inputs
        image = kwargs.get("image")
        images = kwargs.get("images")
        pixel_values = kwargs.get("pixel_values")
        
        # only provide a single image without preprocessing
        if pixel_values is None and image is not None:
            image_processor = self.load_processor()
            images = [image] * batch_size
            pixel_values = image_processor(images, return_tensors="pt")["pixel_values"]
        # only provide a list of images without preprocessing
        elif pixel_values is None and images is not None:
            image_processor = self.load_processor()
            pixel_values = image_processor(images, return_tensors="pt")["pixel_values"]
        # create a random tensor
        elif pixel_values is None:
            pixel_values = torch.randn(batch_size, 3, 224, 224)
        # already provide a preprocessed image tensors (pixel_values)
        elif pixel_values is not None and image is None and images is None:
            pass
        else:
            raise ValueError("Either images or pixel_values should be provided")
        
        if "image" in kwargs: 
            kwargs.pop("image")
        kwargs["pixel_values"] = pixel_values
        return kwargs
    
    def inference(self, batch_size: int, model: PreTrainedModel, **kwargs):
        """
        Base inference function
        """
        set_seed(0)
        kwargs = self.fetch_pixel_values_for_kwargs(batch_size, **kwargs)
        outputs = model(**kwargs)
        return outputs

    def check_accuracy_logits( self, traced_model: PreTrainedModel, batch_size: int, **kwargs):
        """
        Function to compare outputs from huggingface model and neuronx NxD model
        """
        # outputs_expected = self.inference_on_cpu(batch_size, **kwargs)
        outputs_expected = self.inference_with_hf_model(batch_size, **kwargs)
        outputs_actual = self.inference_on_neuron(batch_size, traced_model, **kwargs)

        # check if the logits are close enough
        for atol in [1e-4, 1e-5, 1e-6, 1e-7]:
            is_close = torch.allclose(outputs_expected, outputs_actual, atol=atol)
            if is_close:
                print(f"[PASS] The output from Neuronx NxD is accurate for Absolute tolerance: {atol}!")
            else:
                print(f"[FAIL] The output from Neuronx NxD is NOT accurate for Absolute tolerance: {atol}!")

    def benchmark_sampling(self, model: PreTrainedModel, target: str = None, **kwargs):
        neuron_config = model.neuron_config
        target = target if target is not None else "all"

        report = {}
        # Benchmark image encoding model only
        if target in ["all", "image_encode"]:
            input_param = self.get_sample_inputs(IMAGE_ENCODING_MODEL, neuron_config, **kwargs)
            img_enc_benchmark = Benchmark(model.image_encoding_model, input_param)
            img_enc_benchmark.run()
            report[IMAGE_ENCODING_MODEL] = generate_report(img_enc_benchmark.latency_list, neuron_config.max_batch_size)

        model.reset()

        print("Benchmark completed and its result is as following")
        print(json.dumps(report, indent=4))
        with open(BENCHMARK_REPORT_FILENAME, "w") as f:
            json.dump(report, f)
        print("Completed saving result to " + BENCHMARK_REPORT_FILENAME)

        return report


if __name__ == "__main__":
    ViTRunner.cmd_execute()
