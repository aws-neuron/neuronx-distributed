import pytest
import torch
from neuronx_distributed.modules.moe.moe_configs import RouterConfig

class TestRouterConfig:
    def test_default_initialization(self):
        config = RouterConfig()
        assert config.act_fn == "softmax"
        assert config.dtype == torch.float32

    def test_custom_activation_function(self):
        config = RouterConfig(act_fn="gelu")
        assert config.act_fn == "gelu"
        assert config.dtype == torch.float32

    def test_fp16_dtype(self):
        config = RouterConfig(dtype=torch.float16)
        assert config.dtype == torch.float16

    def test_bf16_dtype(self):
        config = RouterConfig(dtype=torch.bfloat16)
        assert config.dtype == torch.bfloat16

    def test_from_kwargs_defaults(self):
        config = RouterConfig.from_kwargs()
        assert config.act_fn == "softmax"
        assert config.dtype == torch.float32

    def test_from_kwargs_custom_act_fn(self):
        config = RouterConfig.from_kwargs(router_act_fn="relu")
        assert config.act_fn == "relu"
        assert config.dtype == torch.float32

    def test_from_kwargs_dtype_object(self):
        config = RouterConfig.from_kwargs(router_dtype=torch.float16)
        assert config.dtype == torch.float16

    def test_from_kwargs_dtype_string_fp16(self):
        config = RouterConfig.from_kwargs(router_dtype="float16")
        assert config.dtype == torch.float16

    def test_from_kwargs_dtype_string_bf16(self):
        config = RouterConfig.from_kwargs(router_dtype="bfloat16")
        assert config.dtype == torch.bfloat16

    def test_from_kwargs_dtype_string_fp32(self):
        config = RouterConfig.from_kwargs(router_dtype="float32")
        assert config.dtype == torch.float32

    def test_from_kwargs_both_parameters(self):
        config = RouterConfig.from_kwargs(
            router_act_fn="gelu",
            router_dtype=torch.bfloat16
        )
        assert config.act_fn == "gelu"
        assert config.dtype == torch.bfloat16

    def test_from_kwargs_ignores_extra_kwargs(self):
        config = RouterConfig.from_kwargs(
            router_act_fn="relu",
            router_dtype=torch.float16,
            unrelated_param="ignored"
        )
        assert config.act_fn == "relu"
        assert config.dtype == torch.float16

if __name__ == "__main__":
    pytest.main([__file__, "-v"])