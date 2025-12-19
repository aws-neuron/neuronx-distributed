import torch
from typing import Dict, List

class RuntimeRegister:
    """Static registry for runtime tensors and masks keyed by module name."""
    _tr_runtime_list: Dict[str, torch.Tensor] = {}
    _tr_mask_list: Dict[str, torch.Tensor] = {}
    module_superset: List[str] = []
    
    @classmethod
    def register_runtime_args(cls, tr_args:List[torch.Tensor], mask_args:List[torch.Tensor]):
        if len(tr_args) != len(cls.module_superset) or len(mask_args) != len(cls.module_superset):
            raise ValueError(
                f"[TF:] Expected {len(cls.module_superset)} tf tensors & {len(cls.module_superset)} masks, "
                f"got {len(tr_args)} and {len(mask_args)}"
            )

        # Evict any previously registered args
        cls.clear_runtime_args()

        for mod_name, t_src, m_src in zip(cls.module_superset, tr_args, mask_args):
            if not isinstance(t_src, torch.Tensor):
                raise TypeError(f"[TF:] Tensor for '{mod_name}' must be a torch.Tensor, got {type(t_src)}")
            
            t = t_src.clone().detach()
            m = m_src.clone().detach()

            cls._tr_runtime_list[mod_name] = t
            cls._tr_mask_list[mod_name] = m
    
    @classmethod
    def clear_runtime_args(cls):
        cls._tr_runtime_list = {}
        cls._tr_mask_list = {}
