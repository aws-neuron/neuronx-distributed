from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional


@dataclass
class LoraConfig:
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        lora_rank (`int`):
            Lora attention dimension (the "rank").
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings.
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
        lora_dropout (`float`):
            The dropout probability for Lora layers.
        bias (`str`):
            Bias type for LoRA. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        use_rslora (`bool`):
            When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a> which
            sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it was proven to work better.
            Otherwise, it will use the original default value of `lora_alpha/r`.
        init_lora_weights (`Literal["default", "gaussian"]`):
            How to initialize the weights of the adapter layers. Passing 'default' results in the default
            initialization from the reference implementation from Microsoft. Passing 'gaussian' results in Gaussian
            initialization scaled by the LoRA rank for linear and layers. Setting the initialization to False leads to
            completely random initialization and is discouraged.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    """

    enable_lora: bool = field(default=False, metadata={"help": "Will apply LoRA for the model fine-tuning."})
    lora_rank: int = field(default=16, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q_proj', 'v_proj'] for Llama. Refer to constants.py for more examples."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses Rank-Stabilized LoRA doi.org/10.48550/arXiv.2312.03732"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )
    init_lora_weights: Literal["default", "gaussian"] = field(
        default="default",
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. Passing 'default' results in the default "
                "initialization from the reference implementation from Microsoft. Passing 'gaussian' results "
                "in Gaussian initialization scaled by the LoRA rank for linear and layers. Setting the initialization "
                "to False leads to completely random initialization and is discouraged."
            ),
        },
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    lora_verbose: bool = field(default=False, metadata={"help": "print out LoRA information"})
    load_lora_from_ckpt: bool = field(
        default=False, metadata={"help": "Will load LoRA adapter and base model from a checkpoint."}
    )
    lora_load_tag: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Specify the adapter tag to load LoRA checkpoint."
                "If not specified, the latest adapter checkpoint will be used."
            )
        },
    )
    lora_save_dir: Optional[str] = field(
        default="lora_adapter",
        metadata={
            "help": (
                "Specify the folder to load and save LoRA checkpoint."
                "If not specified, the default path will be used."
            )
        },
    )
    merge_lora: bool = field(
        default=False, metadata={"help": "merge the LoRA adapter into the base model for checkpoint"}
    )
    save_lora_base: bool = field(default=False, metadata={"help": "save the base model"})
    save_lora_config_adapter: bool = field(
        default=True, metadata={"help": "save LoRA configuration and LoRA adapter in the same checkpoint file."}
    )

    @staticmethod
    def get_selected_fields():
        return [
            "bias",
            "init_lora_weights",
            "lora_alpha",
            "lora_dropout",
            "lora_rank",
            "use_rslora",
            "target_modules",
            "modules_to_save",
            "save_lora_base",
            "merge_lora",
            "save_lora_config_adapter",
        ]

    def selected_fields_to_save(self):
        config_dict = asdict(self)
        selected_dict = {}
        selected_fields = self.get_selected_fields()

        for key, value in config_dict.items():
            if key in selected_fields:
                selected_dict[key] = value

        return selected_dict

    def __post_init__(self):
        self.target_modules = set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
