# multistep_tf_register.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re
import torch
import torch.nn.functional as F

# Files like:
#   captured_tensors_cte_step_1_module_layers.0.moe.router.pt
#   captured_tensors_tkg_step_2_module_layers.7.attn.qkv_output.pt
_RE = re.compile(r"^captured_tensors_(?:cte|ctx|tkg)_step_(\d+)_module_(.+?)(?:[._]outputs?)?\.pt$")


def _scan(dir_path: Optional[Path]) -> Dict[int, Dict[str, torch.Tensor]]:
    """Return {step: {module: tensor}} for all *.pt files in dir_path."""
    out: Dict[int, Dict[str, torch.Tensor]] = {}

    if dir_path is None:
        raise ValueError("dir_path is None")

    # Accept both str and Path
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    if not dir_path.exists():
        raise ValueError(f"dir_path does not exist: {dir_path}")

    for p in dir_path.glob("*.pt"):
        m = _RE.match(p.name)
        if not m:
            continue

        step = int(m.group(1))
        module = m.group(2)

        try:
            t = torch.load(str(p), map_location="cpu")
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"{p} does not contain a tensor")
        except Exception as e:
            raise Exception(e)
        out.setdefault(step, {})[module] = t
    return out



def _right_pad_to(x: torch.Tensor, target: torch.Size) -> torch.Tensor:
    if x.dim() != len(target):
        raise ValueError(f"rank mismatch: {tuple(x.shape)} vs {tuple(target)}")
    pads = []
    for d in reversed(range(x.dim())):
        need = int(target[d]) - int(x.size(d))
        if need < 0:
            raise ValueError(f"cannot shrink {tuple(x.shape)} to {tuple(target)}")
        pads.extend([0, need])
    return F.pad(x, pads) if any(pads) else x


def _align_to_neuron(cpu_t: Optional[torch.Tensor], neu_t: torch.Tensor) -> torch.Tensor:
    """Make CPU tensor match Neuron tensor rank/shape/dtype (no truncation)."""
    if cpu_t is None:
        return torch.zeros_like(neu_t)
    x = cpu_t
    # If CPU missing a leading batch dim (common), add it (only if Neuron has it)
    if x.dim() == neu_t.dim() - 1:
        x = x.unsqueeze(0)
    # Enforce same rank and right-pad to match each dim
    x = _right_pad_to(x, neu_t.shape)
    # Cast dtype to Neuron’s dtype
    if x.dtype != neu_t.dtype:
        x = x.to(dtype=neu_t.dtype)
    return x


class MultiStepTFRegister:
    """
    Multi-step TF helper with ONLY two public APIs as class methods:

      - example_args(step) -> (rep_list, mask_list)
          zeros with Neuron shapes; masks are scalar bools (True)
      - step_args(step)    -> (rep_list, mask_list)
          aligned CPU tensors (Neuron-shaped) for modules in tr_config[step];
          masks are scalar bools (True).  (Adjust if you need selective masking.)

    Usage:
      MultiStepTFRegister.configure(cpu_dir=Path(...),
                                    neuron_dir=Path(...),
                                    tr_config={step:[mods,...]})
      reps, masks = MultiStepTFRegister.example_args(step=1)
      reps, masks = MultiStepTFRegister.step_args(step=1)
    """

    _tr_config: Dict[int, List[str]] = {}
    _neu_by_step: Dict[int, Dict[str, torch.Tensor]] = {}
    _cpu_by_step: Dict[int, Dict[str, torch.Tensor]] = {}
    _modules: List[str] = []
    _aligned_cpu: Dict[int, Dict[str, torch.Tensor]] = {}

    @classmethod
    def reset(cls):
        cls._tr_config = {}
        cls._neu_by_step = {}
        cls._cpu_by_step = {}
        cls._aligned_cpu = {}
        cls._modules = []

    @classmethod
    def configure(cls,
                  cpu_dir: Path,
                  neuron_dir: Path,
                  tr_config: Dict[int, List[str]]) -> None:
        cls._tr_config = tr_config or {}
        cls._neu_by_step = _scan(neuron_dir)
        cls._cpu_by_step = _scan(cpu_dir)

        # Build module order from tr_config (stable, de-duped)
        order: List[str] = []
        seen = set()
        for s in sorted(cls._tr_config):
            for m in cls._tr_config[s]:
                if m not in seen:
                    seen.add(m)
                    order.append(m)
        cls._modules = order

        # Precompute aligned CPU tensors to Neuron tensors
        cls._aligned_cpu = {}
        for step, neu_map in cls._neu_by_step.items():
            cls._aligned_cpu.setdefault(step, {})
            cpu_map = cls._cpu_by_step.get(step, {})
            for m in cls._modules:
                neu_t = neu_map.get(m)
                if neu_t is None:
                    raise ValueError(f"Missing neuron tensor for step={step} and module={m}")
                cls._aligned_cpu[step][m] = _align_to_neuron(cpu_map.get(m), neu_t) 

    # ----------------- Public API -----------------
    @classmethod
    def example_args(cls, step: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        For tracing: zeros with Neuron shapes; masks = ones_like (valid shape).
        If a module has no Neuron tensor for this step, we synthesize zeros
        from any available CPU tensor for that (step, module); else (1,1,1).
        """
        reps: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        neu_map = cls._neu_by_step.get(step, {})

        for m in cls._modules:
            neu_t = neu_map.get(m)
            z = torch.zeros_like(neu_t)
            reps.append(z)
            masks.append(torch.tensor(True))
        return reps, masks

    @classmethod
    def step_args(cls, step: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        For runtime: provide aligned CPU tensors and masks according to tr_config.
        """
        reps: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        neu_map = cls._neu_by_step.get(step, {})
        aligned = cls._aligned_cpu.get(step, {})

        for m in cls._modules:
            neu_t = neu_map.get(m)
            if neu_t is None:
                raise ValueError(f'Missing neuron tensor for {step} and {m}')

            rep = aligned[m]
            reps.append(rep)
            masks.append(torch.tensor(True))
        return reps, masks
