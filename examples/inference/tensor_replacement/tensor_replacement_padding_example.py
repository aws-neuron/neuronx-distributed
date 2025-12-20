"""Teacher-Forcing (TF) demo with:

Directory structure expected at runtime (created automatically under CWD):
./tensor_replacement/
- traced artifacts, weights, and capture dumps live here


Notes:
• `modify_model_for_tensor_replacement` returns (patched_model, hooks). We unpack.
• TF masks are scalar booleans (torch.tensor(True|False)) per-argument slot.
• CPU-captured tensors may be shorter along the sequence dim; we right‑pad to Neuron.
"""

import os
import json
from functools import partial
from typing import Dict, List, Tuple, Optional
import traceback
from neuronx_distributed.utils.tensor_replacement.registry import RuntimeRegister
import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed.trace import ModelBuilder
from neuronx_distributed.trace.model_builder import BaseModelInstance
from neuronx_distributed.utils.logger import get_logger

# Capture / TF utils
from neuronx_distributed.utils.tensor_capture import (
    enable_tensor_capture,
    get_available_modules,
)
from neuronx_distributed.utils.tensor_replacement import modify_model_for_tensor_replacement

# Viz util (dir-based)
from tr_visualize import visualize_tensor_differences_over_steps
from step_register import MultiStepTFRegister
from toy_model.model import ToyDeepModel

logger = get_logger()

# ---------------------------------------------------------------------------
# Paths & small helpers
# ---------------------------------------------------------------------------

HOME = os.path.join(os.getcwd(), "tensor_replacement")
os.makedirs(HOME, exist_ok=True)
print("Results working directory:", HOME)


def _p(*parts: str) -> str:
    """Join paths under our ROOT folder."""
    return os.path.join(HOME, *parts)


# ---------------------------------------------------------------------------
# ModelInstance for tracing
# ---------------------------------------------------------------------------

class TFMultiLayerInstance(BaseModelInstance):
    """Builds a `ToyDeepModel` with optional features:


    - **Tensor Replacement (Teacher Forcing)** via `modify_model_for_tensor_replacement`.
    When `tr_config` is provided, we also configure a global module order for
    the patched forward using `RuntimeRegister.module_superset` (so Neuron knows
    in which order replacement tensors and masks are passed).


    - **Tensor Capture** for a provided list of module names, so intermediate
    activations are dumped to disk for later visualization / TF injection.


    Parameters
    ----------
    hidden_size : int
    Model hidden size.
    intermediate_size : int
    MLP/FFN intermediate (expansion) size.
    modules_to_capture : list[str] | None
    List of module qualified names to hook for capture.
    max_tensors : int | None
    Max number of tensors to retain per module (capture util param).
    tr_config : dict[int, list[str]] | None
    Per-step map of modules to be replaced (teacher forcing); if None, TF is off.
    cpu_dir, neuron_dir : str | None
    Directories with CPU/Neuron capture files (used by MultiStepTFRegister).
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        modules_to_capture: Optional[List[str]] = None,
        max_tensors: Optional[int] = None,
        tr_config: Optional[Dict[int, List[str]]] = None,
        cpu_dir: Optional[str] = None,
        neuron_dir: Optional[str] = None,
    ):
        self.module: Optional[nn.Module] = None
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.modules_to_capture = modules_to_capture or []
        self.max_tensors = max_tensors
        self.tr_config = tr_config
        self.cpu_dir = cpu_dir
        self.neuron_dir = neuron_dir

        # Configure the classmethod-only TF register once if TF is requested.
        # The register loads captures and derives module ordering used by RuntimeRegister.
        if self.tr_config:
            MultiStepTFRegister.configure(
            cpu_dir=self.cpu_dir,
            neuron_dir=self.neuron_dir,
            tr_config=self.tr_config,
            )
            # The patched forward consumes args in this superset order.
            RuntimeRegister.module_superset = MultiStepTFRegister._modules

    def load_module(self):
        model = ToyDeepModel(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
        )

        # Enable tensor replacement (teacher forcing) if configured
        if self.tr_config:
            logger.info(f"[TF] Enabling tensor replacement for modules map: {self.tr_config}")
            model, _ = modify_model_for_tensor_replacement(model)

        # Enable tensor capture hooks if requested
        if self.modules_to_capture:
            logger.info(f"[TC] Enabling tensor capture for: {self.modules_to_capture}")
            model = enable_tensor_capture(model, self.modules_to_capture, self.max_tensors)

        self.module = model

    def get(self, bucket_rank, **kwargs):
        return self.module, {}
    

def compile_and_save(
    output_path,
    jitter=439,
    modules_to_capture=None,
    max_tensors=100,
    tp_degree=1,
    hidden_size=2048,
    intermediate_size=8192,
    tr_config=None,
    tr_cpu_dir=None,
    tr_neuron_dir=None
):
    
    """
    Compile CTX (seq=128) and TKG (seq=1) programs into a single traced artifact.
    Emits: <HOME>/<output_path>/{model.pt, weights/, config.json}
    """
    # --- seeding & dirs
    torch.manual_seed(42 + jitter)
    out_dir = os.path.join(HOME, output_path)
    os.makedirs(out_dir, exist_ok=True)

    # --- temp model just to save weights & validate capture names
    temp_model = ToyDeepModel(hidden_size=hidden_size, intermediate_size=intermediate_size)
    weights_path = os.path.join(out_dir, "weights.pt")
    torch.save(temp_model.state_dict(), weights_path)
    print(f"Model weights saved to {out_dir}")

    # Validate capture module names (fail fast)
    if modules_to_capture is None:
        modules_to_capture = [
            "layers.0.moe.router",
            "layers.1.moe.router",
            "layers.2.moe.router",
            "layers.3.moe.router",
        ]
    available_modules = set(get_available_modules(temp_model))
    logger.info(f"Available modules: {available_modules}")
    missing = [m for m in modules_to_capture if m not in available_modules]
    if missing:
        raise ValueError(f"Modules not present in model: {missing}")

    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=partial(torch.load, weights_path),
        debug=True
    )

    # Small helpers
    def _make_instance():
        return TFMultiLayerInstance(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            modules_to_capture=modules_to_capture,
            max_tensors=max_tensors,
            tr_config=tr_config,
            cpu_dir=tr_cpu_dir,
            neuron_dir=tr_neuron_dir,
        )

    def _example_inputs(seq_len: int, tr_step: int | None):
        """Build example inputs for the tracer.


        If TF is enabled, we ask the classmethod-only register for shape-correct
        dummy tensors and scalar-bool masks for the requested step.
        """
        ex = []
        if tr_config and tr_step is not None:
            tr_tensors, tr_masks = MultiStepTFRegister.example_args(step=tr_step)
            ex.append((torch.randn(1, seq_len, hidden_size), *tr_tensors, *tr_masks))
        else:
            ex.append((torch.randn(1, seq_len, hidden_size),))
        return ex
    
    # Add CTX (seq=128) and TKG (seq=1) programs
    ctx_ins = _make_instance()
    builder.add(
        key="ctx",
        model_instance=ctx_ins,
        example_inputs=_example_inputs(seq_len=128, tr_step=1 if tr_config else None),
        compiler_args="--auto-cast=none",
    )
    tkg_ins = _make_instance()
    builder.add(
        key="tkg",
        model_instance=tkg_ins,
        example_inputs=_example_inputs(seq_len=1, tr_step=2 if tr_config else None),
        compiler_args="--auto-cast=none",
    )

    logger.info("Compiling TwoLayerMLP with selective tensor replacement")
    # --- trace & save
    try:
        traced = builder.trace(initialize_model_weights=True)
        builder.shard_checkpoint(serialize_path=os.path.join(out_dir, "weights"))
        torch.jit.save(traced, os.path.join(out_dir, "model.pt"))

        # Save a config file with model information 
        config = { "hidden_size": hidden_size, "intermediate_size": intermediate_size, "batch_sizes": [1], "tp_degree": tp_degree, "captured_modules": modules_to_capture, "max_tensors": max_tensors, "tr_config": tr_config }
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model compiled and saved to {output_path}")

        # show attached hooks once, from the last instance
        logger.info("Hooks on ctx model")
        for name, module in ctx_ins.module.named_modules():
            hooks = list(getattr(module, "_forward_hooks", {}).items())
            if hooks:
                print(f"Module: {name}")
                for hook_id, hook_fn in hooks:
                    print(f"  Hook ID: {hook_id} | Hook: {hook_fn}")

        logger.info("Hooks on tkg model")
        for name, module in tkg_ins.module.named_modules():
            hooks = list(getattr(module, "_forward_hooks", {}).items())
            if hooks:
                print(f"Module: {name}")
                for hook_id, hook_fn in hooks:
                    print(f"  Hook ID: {hook_id} | Hook: {hook_fn}")

        return out_dir
    except Exception as e:
        logger.error(f"Error during model tracing: {e}")
        traceback.print_exc()
        return None



@torch.inference_mode()
def run_neuron_inference(
    neuron_model_path="traced_model_with_tc_neuron",
    cpu_model_path="traced_model_with_tc_cpu",
    tc_save="tc_model_",
    run_cpu=True,
    tensor_replace=False,
    num_gen_steps=5,
    ctx_seq_len_cpu=48,
    ctx_seq_len_neuron=128,
):
    """
    Runs a ctx pass then `num_gen_steps` token-gen steps for both Neuron and CPU.
    Saves captured tensors to `tc_save + 'neuron'` and `tc_save + 'cpu'`.

    Returns:
      (neuron_results, cpu_results)  # lists of dicts in chronological order
      If run_cpu=False: returns (neuron_results,)
    """
    logger.info("Running Neuron inference (ctx + tkg)")

    # ---------- small helpers ----------
    def _right_pad_T(x: torch.Tensor, target_T: int) -> torch.Tensor:
        T = x.size(1)
        if T >= target_T:
            return x[:, :target_T, :]
        return F.pad(x, (0, 0, 0, target_T - T))  # pad length on dim=1 (seq)

    def _load_config(model_dir: str):
        with open(os.path.join(HOME, model_dir, "config.json"), "r") as f:
            return json.load(f)

    def _load_weights(weights_dir: str, tp_degree: int):
        w = []
        if not os.path.isdir(weights_dir):
            return w
        # prefer safetensors if present; fall back to .pt
        try:
            from safetensors.torch import load_file
            for rank in range(tp_degree):
                p = os.path.join(weights_dir, f"tp{rank}_sharded_checkpoint.safetensors")
                if os.path.exists(p):
                    w.append(load_file(p))
        except Exception:
            pass
        if not w:  # .pt fallback or mixed
            for rank in range(tp_degree):
                p = os.path.join(weights_dir, f"tp{rank}_sharded_checkpoint.pt")
                if os.path.exists(p):
                    w.append(torch.load(p))
        return w

    def _init_neuron_model(dir_name: str, tp_degree: int):
        m = torch.jit.load(os.path.join(HOME, dir_name, "model.pt"))
        w = _load_weights(os.path.join(HOME, dir_name, "weights"), tp_degree)
        start_rank = torch.tensor([0], dtype=torch.int32, device="cpu")
        m.nxd_model.initialize(w, start_rank)
        return m

    def _ensure_cpu_model(cpu_dir: str, captured_modules, max_tensors):
        model = ToyDeepModel()
        try:
            wpt = os.path.join(HOME, cpu_dir, "weights.pt")
            if os.path.exists(wpt):
                model.load_state_dict(torch.load(wpt))
        except Exception as e:
            logger.warning(f"[CPU] load_state_dict failed: {e}")
        return enable_tensor_capture(model, captured_modules, max_tensors)

    def _save_tensor_dict_to_dir(tensor_dict, save_dir, step: int):
        abs_dir = os.path.join(HOME, save_dir)
        os.makedirs(abs_dir, exist_ok=True)
        for k, v in tensor_dict.items():
            key = f"captured_tensors_{'cte' if step == 1 else 'tkg'}_step_{step}_module_{k}"
            path = os.path.join(abs_dir, f"{key}.pt")
            torch.save(v, path)


    # ---------- load compile-time info ----------
    try:
        cfg = _load_config(neuron_model_path)
        hidden_size = cfg.get("hidden_size", 2048)
        tp_degree = cfg.get("tp_degree", 1)
        captured_modules = cfg.get("captured_modules", [])
        max_tensors = cfg.get("max_tensors", 100)

        # Neuron model (one-time)
        neuron_model = _init_neuron_model(neuron_model_path, tp_degree)
        logger.info("Neuron model initialized")

        # CPU model (one-time)
        cpu_model = None
        if run_cpu:
            cpu_model = _ensure_cpu_model(cpu_model_path, captured_modules, max_tensors)

        # results stored in chronological order:
        #   index 0 = CTX, then TKG steps 1..N
        neuron_results = []
        cpu_results = [] if run_cpu else None

        # For each compiled batch size, run the full ctx+tkg sequence
        for batch_size in cfg.get("batch_sizes", [1]):
            logger.info(f"[Batch {batch_size}] CTX + {num_gen_steps} TKG steps")

            # --- build ctx inputs (same base, neuron right-padded to 128) ---
            torch.manual_seed(42)
            ctx_cpu = torch.randn(batch_size, ctx_seq_len_cpu, hidden_size)
            ctx_neu = _right_pad_T(ctx_cpu, ctx_seq_len_neuron)

            # --- NEURON: CTX pass ---
            if tensor_replace:
                tf_list, mask_list = MultiStepTFRegister.step_args(step=1)  # CTX = step 1
                out_neu_ctx = neuron_model(ctx_neu, *tf_list, *mask_list)
            else:
                out_neu_ctx = neuron_model(ctx_neu)
            neu_ctx_dict = out_neu_ctx[1]
            _save_tensor_dict_to_dir(neu_ctx_dict, tc_save + "neuron", step=1)
            neuron_results.append(neu_ctx_dict)

            # --- CPU: CTX pass ---
            if run_cpu:
                out_cpu_ctx = cpu_model(ctx_cpu)
                cpu_ctx_dict = out_cpu_ctx[1]
                _save_tensor_dict_to_dir(cpu_ctx_dict, tc_save + "cpu", step=1)
                cpu_results.append(cpu_ctx_dict)

            # --- token-gen steps ---
            for s in range(1, num_gen_steps + 1):
                torch.manual_seed(1000 + s)
                tok = torch.randn(batch_size, 1, hidden_size)

                # neuron tkg
                if tensor_replace:
                    tf_list, mask_list = MultiStepTFRegister.step_args(step=s + 1)  # tkg steps are 2..N+1
                    out_neu_tkg = neuron_model(tok, *tf_list, *mask_list)
                else:
                    out_neu_tkg = neuron_model(tok)
                neu_tkg_dict = out_neu_tkg[1]
                _save_tensor_dict_to_dir(neu_tkg_dict, tc_save + "neuron", step=s + 1)
                neuron_results.append(neu_tkg_dict)

                # cpu tkg
                if run_cpu:
                    out_cpu_tkg = cpu_model(tok)
                    cpu_tkg_dict = out_cpu_tkg[1]
                    _save_tensor_dict_to_dir(cpu_tkg_dict, tc_save + "cpu", step=s + 1)
                    cpu_results.append(cpu_tkg_dict)

        MultiStepTFRegister.reset()
        return (neuron_results, cpu_results) if run_cpu else (neuron_results,)

    except Exception as e:
        MultiStepTFRegister.reset()
        logger.error(f"Error during inference: {e}")
        traceback.print_exc()



if __name__ == "__main__":
    # Using different weights (jitter arg) for cpu vs neuron so intermediate outputs are different
    cpu_model = compile_and_save(output_path='traced_model_with_tc_cpu', jitter=440) # with TensorCapture for CPU
    neu_model = compile_and_save(output_path='traced_model_with_tc_neuron') # with TensorCapture enabled, to be run on neuron
    neuron_results, cpu_results = run_neuron_inference(neuron_model_path=neu_model, cpu_model_path=cpu_model) # with TensorCapture on CPU and Neuron

    # Visualize the differences between cpu and neuron captured tensors per generation step on a scatter plot
    visualize_tensor_differences_over_steps(dir_cpu=f'{HOME}/tc_model_cpu', dir_neuron=f'{HOME}/tc_model_neuron')

    print("\n=== Second run with teacher forcing and tensor capture ===\n\n\n")

    tr_config = {
        1:   ["layers.0.moe.router", "layers.1.moe.router", "layers.2.moe.router", "layers.3.moe.router"],
        2:   ["layers.0.moe.router", "layers.1.moe.router", "layers.2.moe.router", "layers.3.moe.router"],
        3:   ["layers.0.moe.router", "layers.1.moe.router", "layers.2.moe.router", "layers.3.moe.router"],
        4:   ["layers.0.moe.router", "layers.1.moe.router", "layers.2.moe.router", "layers.3.moe.router"],
    }
    
    neu_tr_model = compile_and_save(output_path='traced_model_with_tc_tr_neuron', tr_config=tr_config, tr_cpu_dir=f"{HOME}/tc_model_cpu/", tr_neuron_dir=f"{HOME}/tc_model_neuron/") # with TC and TF
    neuron_tr_tensor_dict, = run_neuron_inference(run_cpu=False, neuron_model_path=neu_tr_model, tc_save='tc_tr_model_', tensor_replace=True) # with TF and TC
    # Visualize the differences between cpu and neuron captured tensors per generation step on a scatter plot
    visualize_tensor_differences_over_steps(dir_cpu=f'{HOME}/tc_model_cpu', dir_neuron=f"{HOME}/tc_tr_model_neuron")



'''
PHASE STEP  MODULE                                                                                          CPU SHAPE              NEURON SHAPE  STATUS
-------------------------------------------------------------------------------------------------------------------------------------------------------
ctx     1  layers.0.attn.q_proj.outputs                                                                (1, 48, 2048)            (1, 128, 2048)  SHAPE MISMATCH
ctx     1  layers.0.mlp.down_proj.outputs                                                              (1, 48, 2048)            (1, 128, 2048)  SHAPE MISMATCH
ctx     1  layers.0.mlp.up_proj.outputs                                                                (1, 48, 4096)            (1, 128, 4096)  SHAPE MISMATCH
ctx     1  layers.0.moe.router.outputs                                                                    (1, 48, 4)               (1, 128, 4)  SHAPE MISMATCH
ctx     1  layers.2.moe.router.outputs                                                                    (1, 48, 4)               (1, 128, 4)  SHAPE MISMATCH
ctx     1  layers.3.attn.q_proj.outputs                                                                (1, 48, 2048)            (1, 128, 2048)  SHAPE MISMATCH
ctx     1  layers.3.mlp.up_proj.outputs                                                               (1, 48, 16384)           (1, 128, 16384)  SHAPE MISMATCH
ctx     1  layers.4.attn.outputs                                                                       (1, 48, 2048)            (1, 128, 2048)  SHAPE MISMATCH
tkg     2  layers.0.attn.q_proj.outputs                                                                 (1, 1, 2048)              (1, 1, 2048)  OK
tkg     2  layers.0.mlp.down_proj.outputs                                                               (1, 1, 2048)              (1, 1, 2048)  OK
tkg     2  layers.0.mlp.up_proj.outputs                                                                 (1, 1, 4096)              (1, 1, 4096)  OK
tkg     2  layers.0.moe.router.outputs                                                                     (1, 1, 4)                 (1, 1, 4)  OK
tkg     2  layers.2.moe.router.outputs                                                                     (1, 1, 4)                 (1, 1, 4)  OK
tkg     2  layers.3.attn.q_proj.outputs                                                                 (1, 1, 2048)              (1, 1, 2048)  OK
tkg     2  layers.3.mlp.up_proj.outputs                                                                (1, 1, 16384)             (1, 1, 16384)  OK
tkg     2  layers.4.attn.outputs                                                                        (1, 1, 2048)              (1, 1, 2048)  OK
tkg     3  layers.0.attn.q_proj.outputs                                                                 (1, 1, 2048)              (1, 1, 2048)  OK
tkg     3  layers.0.mlp.down_proj.outputs                                                               (1, 1, 2048)              (1, 1, 2048)  OK
tkg     3  layers.0.mlp.up_proj.outputs                                                                 (1, 1, 4096)              (1, 1, 4096)  OK
tkg     3  layers.0.moe.router.outputs                                                                     (1, 1, 4)                 (1, 1, 4)  OK
tkg     3  layers.2.moe.router.outputs                                                                     (1, 1, 4)                 (1, 1, 4)  OK
tkg     3  layers.3.attn.q_proj.outputs                                                                 (1, 1, 2048)              (1, 1, 2048)  OK
tkg     3  layers.3.mlp.up_proj.outputs                                                                (1, 1, 16384)             (1, 1, 16384)  OK
tkg     3  layers.4.attn.outputs                                                                        (1, 1, 2048)              (1, 1, 2048)  OK
tkg     4  layers.0.attn.q_proj.outputs                                                                 (1, 1, 2048)              (1, 1, 2048)  OK
tkg     4  layers.0.mlp.down_proj.outputs                                                               (1, 1, 2048)              (1, 1, 2048)  OK
tkg     4  layers.0.mlp.up_proj.outputs                                                                 (1, 1, 4096)              (1, 1, 4096)  OK
tkg     4  layers.0.moe.router.outputs                                                                     (1, 1, 4)                 (1, 1, 4)  OK
tkg     4  layers.2.moe.router.outputs                                                                     (1, 1, 4)                 (1, 1, 4)  OK
tkg     4  layers.3.attn.q_proj.outputs                                                                 (1, 1, 2048)              (1, 1, 2048)  OK
tkg     4  layers.3.mlp.up_proj.outputs                                                                (1, 1, 16384)             (1, 1, 16384)  OK
tkg     4  layers.4.attn.outputs                                                                        (1, 1, 2048)              (1, 1, 2048)  OK
tkg     5  layers.0.attn.q_proj.outputs                                                                 (1, 1, 2048)              (1, 1, 2048)  OK
tkg     5  layers.0.mlp.down_proj.outputs                                                               (1, 1, 2048)              (1, 1, 2048)  OK
tkg     5  layers.0.mlp.up_proj.outputs                                                                 (1, 1, 4096)              (1, 1, 4096)  OK
tkg     5  layers.0.moe.router.outputs                                                                     (1, 1, 4)                 (1, 1, 4)  OK
tkg     5  layers.2.moe.router.outputs                                                                     (1, 1, 4)                 (1, 1, 4)  OK
tkg     5  layers.3.attn.q_proj.outputs                                                                 (1, 1, 2048)              (1, 1, 2048)  OK
tkg     5  layers.3.mlp.up_proj.outputs                                                                (1, 1, 16384)             (1, 1, 16384)  OK
tkg     5  layers.4.attn.outputs                                                                        (1, 1, 2048)              (1, 1, 2048)  OK
tkg     6  layers.0.attn.q_proj.outputs                                                                 (1, 1, 2048)              (1, 1, 2048)  OK
tkg     6  layers.0.mlp.down_proj.outputs                                                               (1, 1, 2048)              (1, 1, 2048)  OK
tkg     6  layers.0.mlp.up_proj.outputs                                                                 (1, 1, 4096)              (1, 1, 4096)  OK
tkg     6  layers.0.moe.router.outputs                                                                     (1, 1, 4)                 (1, 1, 4)  OK
tkg     6  layers.2.moe.router.outputs                                                                     (1, 1, 4)                 (1, 1, 4)  OK
tkg     6  layers.3.attn.q_proj.outputs                                                                 (1, 1, 2048)              (1, 1, 2048)  OK
tkg     6  layers.3.mlp.up_proj.outputs                                                                (1, 1, 16384)             (1, 1, 16384)  OK
tkg     6  layers.4.attn.outputs                                                                        (1, 1, 2048)              (1, 1, 2048)  OK
'''