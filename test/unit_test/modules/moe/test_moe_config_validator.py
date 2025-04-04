"""
Unit tests for MoE configuration validation.

Tests various combinations of MoE configurations including:

- Dropless mode settings
- Capacity factor values
- Activation function compatibility
- GLU MLP requirements
- HuggingFace and Megatron model support
"""

import pytest
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pytest import LogCaptureFixture
from neuronx_distributed.modules.moe.moe_config_validator import MoeConfigValidator


class TestConstants:
    HF = "hf"
    MEGATRON = "megatron"
    DBRX = "dbrx"
    SILU = "silu"
    GELU = "gelu"
    SWIGLU = "swiglu"
    MIXTRAL = "mixtral"


@dataclass
class MoeConfig:
    dropless: bool = False
    capacity_factor: float = 1.0
    glu_mlp: bool = True


@dataclass
class ModelConfig:
    activation: Optional[str] = None
    model_config: str = "config.json"


@dataclass
class Config:
    model_source: str
    model: ModelConfig

    def __post_init__(self):
        self.model.moe = MoeConfig()


class TestMoeConfigValidator:
    @pytest.fixture
    def base_config(self):
        return Config(model_source=TestConstants.HF, model=ModelConfig())

    @pytest.fixture
    def mock_config(self):
        return {"hidden_act": TestConstants.SILU, "model_type": TestConstants.MIXTRAL}

    @pytest.mark.parametrize(
        "test_case",
        [
            pytest.param(
                {
                    "model_source": TestConstants.HF,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": True,
                    "capacity_factor": None,
                    "activation": "silu",
                    "glu_mlp": True,
                    "should_pass": True,
                    "expected_message": "Dropless mode expects a capacity_factor set to 0.0. Current value: None. Setting capacity_factor to 0.0.",
                    "expected_capacity": 0.0,  # Will be set to 0.0 even though initially None
                },
                id="dropless_none_capacity_sets_to_zero",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.HF,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": True,
                    "capacity_factor": 0.0,
                    "activation": "silu",
                    "glu_mlp": True,
                    "should_pass": True,
                    "expected_message": None,
                    "expected_capacity": 0.0,
                },
                id="dropless_zero_capacity_silu_glu_valid",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.HF,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": True,
                    "capacity_factor": 1.0,
                    "activation": "silu",
                    "glu_mlp": True,
                    "should_pass": True,
                    "expected_message": "Dropless mode expects a capacity_factor set to 0.0. Current value: 1.0. Setting capacity_factor to 0.0.",
                    "expected_capacity": 0.0,
                },
                id="dropless_positive_capacity_silu_glu_valid",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.HF,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": True,
                    "capacity_factor": 1.0,
                    "activation": "gelu",
                    "glu_mlp": True,
                    "should_pass": False,
                    "expected_message": "Dropless mode is only supported with SiLU activation function",
                    "expected_capacity": 1.0,
                },
                id="dropless_with_non_silu_activation_invalid",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.HF,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": True,
                    "capacity_factor": 1.0,
                    "activation": "silu",
                    "glu_mlp": False,
                    "should_pass": False,
                    "expected_message": "Dropless mode requires GLU_MLP to be True",
                    "expected_capacity": 1.0,
                },
                id="dropless_silu_no_glu_invalid",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.HF,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": False,
                    "capacity_factor": 0.0,
                    "activation": "gelu",
                    "glu_mlp": True,
                    "should_pass": False,
                    "expected_message": "Dropping requires a capacity factor greater than 0.0",
                    "expected_capacity": 0.0,
                },
                id="dropping_zero_capacity_invalid",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.HF,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": False,
                    "capacity_factor": 1.0,
                    "activation": "gelu",
                    "glu_mlp": True,
                    "should_pass": True,
                    "expected_message": None,
                    "expected_capacity": 1.0,
                },
                id="dropping_positive_capacity_valid",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.HF,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": False,
                    "capacity_factor": None,
                    "activation": "gelu",
                    "glu_mlp": True,
                    "should_pass": True,
                    "expected_message": None,
                    "expected_capacity": None,  # Stays None when dropless is False
                },
                id="dropping_none_capacity_valid",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.HF,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": True,
                    "capacity_factor": -1.5,
                    "activation": "silu",
                    "glu_mlp": True,
                    "should_pass": True,
                    "expected_message": None,
                    "expected_capacity": -1.5,
                },
                id="dropless_negative_capacity_valid",
            ),
            # Megatron test cases
            pytest.param(
                {
                    "model_source": TestConstants.MEGATRON,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": True,
                    "capacity_factor": None,
                    "activation": "silu",
                    "glu_mlp": True,
                    "should_pass": True,
                    "expected_message": "Dropless mode expects a capacity_factor set to 0.0. Current value: None. Setting capacity_factor to 0.0.",
                    "expected_capacity": 0.0,
                },
                id="megatron_dropless_none_capacity_sets_to_zero",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.MEGATRON,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": True,
                    "capacity_factor": 1.0,
                    "activation": "gelu",
                    "glu_mlp": True,
                    "should_pass": False,
                    "expected_message": "For Megatron models, dropless mode is only supported with SiLU or SwiGLU activation function",
                    "expected_capacity": 1.0,
                },
                id="megatron_dropless_gelu_invalid",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.MEGATRON,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": True,
                    "capacity_factor": 0.0,
                    "activation": "silu",
                    "glu_mlp": False,
                    "should_pass": False,
                    "expected_message": "Dropless mode requires GLU_MLP to be True",
                    "expected_capacity": 0.0,
                },
                id="megatron_dropless_no_glu_invalid",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.MEGATRON,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": False,
                    "capacity_factor": 0.0,
                    "activation": "gelu",
                    "glu_mlp": True,
                    "should_pass": False,
                    "expected_message": "Dropping requires a capacity factor greater than 0.0",
                    "expected_capacity": 0.0,
                },
                id="megatron_dropping_zero_capacity_invalid",
            ),
            pytest.param(
                {
                    "model_source": TestConstants.MEGATRON,
                    "model_type": TestConstants.MIXTRAL,
                    "dropless": False,
                    "capacity_factor": 1.0,
                    "activation": "gelu",
                    "glu_mlp": True,
                    "should_pass": True,
                    "expected_message": None,
                    "expected_capacity": 1.0,
                },
                id="megatron_dropping_positive_capacity_valid",
            ),
        ],
    )
    def test_moe_configuration_validation(self, mocker, test_case, caplog):
        """
        Test MoE configurations with different combinations.

        Customer Behavior matrix:
        | Dropless | Capacity Factor| Activation | GLU_MLP | Behavior                                   |
        |----------|----------------|------------|---------|--------------------------------------------|
        | TRUE     | None           | SiLU       | TRUE    | Sets capacity_factor=0.0 & Uses blockwise  |
        | TRUE     | <=0            | SiLU       | TRUE    | Uses blockwise NKI kernel                  |
        | TRUE     | >0             | SiLU       | TRUE    | Sets capacity_factor=0.0  & Uses blockwise |
        | TRUE     | ANY            | NOT SiLU   | ANY     | Raises ValueError         |
        | TRUE     | ANY            | SiLU       | FALSE   | Raises ValueError         |
        | FALSE    | None           | ANY        | ANY     | Uses all experts          |
        | FALSE    | 0              | ANY        | ANY     | Raises ValueError         |
        | FALSE    | >0             | ANY        | ANY     | Uses capacity_factor      |

        Args:
            mocker: pytest-mock fixture for mocking/patching objects during testing
            test_case: Dictionary containing test configuration parameters like:

                - model_source: Source of the model (e.g., "hf", "megatron")
                - activation: Activation function type
                - dropless: Boolean for dropless configuration
                - glu_mlp: Boolean for GLU MLP setting
                - capacity_factor: MoE capacity factor value
                - should_pass: Boolean indicating if test should pass
                - expected_message: Expected log/error message
                - expected_capacity: Expected final capacity factor value
            caplog: pytest fixture for capturing and asserting log messages
        """

        cfg = Config(
            model_source=test_case.get("model_source", TestConstants.HF),
            model=ModelConfig(activation=test_case["activation"]),
        )
        cfg.model.moe.dropless = test_case["dropless"]
        cfg.model.moe.glu_mlp = test_case["glu_mlp"]
        cfg.model.moe.capacity_factor = test_case["capacity_factor"]

        validator = MoeConfigValidator(cfg)

        if test_case.get("model_source") == TestConstants.MEGATRON:
            cfg.model.activation = test_case["activation"]
        else:
            if test_case["model_type"] == TestConstants.DBRX:
                mock_config = {
                    "model_type": TestConstants.DBRX,
                    "ffn_config": {"ffn_act_fn": {"name": test_case["activation"]}},
                }
            else:
                mock_config = {
                    "model_type": TestConstants.MIXTRAL,  # or any non-dbrx value
                    "hidden_act": test_case["activation"],
                }
            mocker.patch.object(validator, "_load_hf_config", return_value=mock_config)

        # Test execution and verification
        if test_case["expected_message"] and test_case["should_pass"]:
            validator.validate_moe_config()
            assert test_case["expected_message"] in caplog.text
        elif test_case["should_pass"]:
            validator.validate_moe_config()
        else:
            with pytest.raises(ValueError, match=test_case["expected_message"]):
                validator.validate_moe_config()

        # Verify final capacity factor value
        if test_case["expected_capacity"] is not None:
            assert cfg.model.moe.capacity_factor == test_case["expected_capacity"]
        else:
            assert cfg.model.moe.capacity_factor is None

    def test_no_moe_config(self):
        """Test that validator raises error when MoE configuration is missing.

        When a configuration without MoE settings is passed to the validator,
        it should raise an AttributeError indicating that MoE configuration
        is required.
        """
        cfg = Config(model_source=TestConstants.HF, model=ModelConfig())
        delattr(cfg.model, "moe")
        validator = MoeConfigValidator(cfg)

        with pytest.raises(AttributeError, match="MoE configuration is missing in model config"):
            validator.validate_moe_config()
