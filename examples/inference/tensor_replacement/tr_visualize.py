import torch
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from typing import Dict
import plotly.io as pio
import re

pio.renderers.default = "browser"

def flatten_tensor(tensor, max_elements=4096):
    if tensor.ndim > 2:
        tensor = tensor[0]
    if tensor.ndim > 2:
        tensor = tensor[0]
    return tensor.flatten()[:max_elements].cpu()


def plot_difference_visualizations_from_flatdicts(
    dict1: dict,
    dict2: dict,
    max_elements=4096,
    base_marker_size=2,
    tkg_boost=False,           # set True for TKG step figures
    phase: str = None,         # e.g., "CTX" or "TKG"
    step: int | None = None,   # e.g., 1-based step index for TKG
):
    common_keys = sorted(set(dict1.keys()) & set(dict2.keys()))
    ncols = 2
    nrows = math.ceil(len(common_keys) / ncols)

    title_main = "Tensor Differences (CPU vs Neuron)"
    if phase and (step is not None):
        title_main = f"{phase} — step {step} · {title_main}"
    elif phase:
        title_main = f"{phase} · {title_main}"

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"{key} - Scatter" for key in common_keys],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    def flatten_tensor(t: torch.Tensor, limit: int) -> torch.Tensor:
        t = t.detach()
        if t.is_sparse:
            t = t.to_dense()
        t = t.reshape(-1).float().cpu()
        return t[:limit] if t.numel() > limit else t

    for i, key in enumerate(common_keys):
        row = i // ncols + 1
        col = i % ncols + 1

        tensor1 = flatten_tensor(dict1[key], max_elements)
        tensor2 = flatten_tensor(dict2[key], max_elements)

        # align shapes (Neuron CTX often right-padded)
        if tensor1.shape != tensor2.shape:
            m = min(tensor1.numel(), tensor2.numel())
            if m == 0:
                print(f"Skipping {key}: empty after alignment")
                continue
            tensor1 = tensor1[:m]
            tensor2 = tensor2[:m]

        # Dynamic marker sizing for tiny TKG tensors
        n_points = min(tensor1.numel(), tensor2.numel())
        ms = base_marker_size
        if tkg_boost:
            ms = max(ms, 6)
        if n_points <= 64:
            ms = max(ms, 8)
        if n_points <= 8:
            ms = max(ms, 12)
        if n_points <= 2:
            ms = max(ms, 16)
        marker_line = dict(width=1) if n_points <= 8 else None

        fig.add_trace(go.Scattergl(
            x=tensor1.numpy(),
            y=tensor2.numpy(),
            mode='markers',
            marker=dict(size=ms, opacity=0.6, line=marker_line),
            name=key
        ), row=row, col=col)

        min_val = min(tensor1.min().item(), tensor2.min().item())
        max_val = max(tensor1.max().item(), tensor2.max().item())
        if min_val == max_val:
            min_val -= 1.0
            max_val += 1.0

        fig.add_shape(
            type='line',
            x0=min_val, y0=min_val, x1=max_val, y1=max_val,
            line=dict(color='red', dash='dash'),
            row=row, col=col
        )

    # Top-level labels
    # Put the main title at the top-left, and create extra top margin for it.
    fig.update_layout(
        height=350 * nrows,
        width=1200,
        title=dict(
            text=title_main,
            x=0.02, xanchor="left",
            y=0.99, yanchor="top",
            font=dict(size=18)
        ),
        showlegend=False,
        margin=dict(t=120, l=60, r=40, b=50),   # more top space prevents collision
    )

    # Optional subtitle: place it just below the title, above the plotting area
    if phase:
        tag = f"<b>Phase:</b> {phase}"
        if step is not None:
            tag += f" &nbsp; &nbsp; <b>Step:</b> {step}"
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=1.02,               # below the title, above the plot
            xanchor="left", yanchor="bottom",
            text=tag,
            showarrow=False,
            align="left",
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0.06)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            borderpad=4,
        )

    fig.show()


_fname_re = re.compile(
    r"captured_tensors_(cte|ctx|tkg)_step_(\d+)_module_(.+)\.pt(?:\.pt)?$"
)

def _safe_load_pt(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None

def load_ctx_tkg_from_dir(root_dir: str):
    """
    Loads a directory of per-tensor files saved like:
      captured_tensors_cte_step_1_module_<module.key>.pt
      captured_tensors_tkg_step_<N>_module_<module.key>.pt
    Returns: {"cte": dict, "tkg": [dict_for_step1, dict_for_step2, ...]}
    """
    ctx_dict = {}
    tkg_steps = {}  # step -> dict

    for fn in os.listdir(root_dir):
        if fn == "capture_metadata.json":
            continue
        m = _fname_re.match(fn)
        if not m:
            continue

        phase, step_s, key = m.groups()
        step = int(step_s)
        full = os.path.join(root_dir, fn)
        obj = _safe_load_pt(full)
        if obj is None or not isinstance(obj, torch.Tensor):
            continue

        if phase == "cte":
            # convention: ctx is step 1
            ctx_dict[key] = obj
        else:
            tkg_steps.setdefault(step, {})[key] = obj

    # Order tkg by ascending step; normalize to a dense list starting at the smallest step
    if tkg_steps:
        ordered_steps = sorted(tkg_steps.keys())
        tkg_list = [tkg_steps[s] for s in ordered_steps]
    else:
        tkg_list = []

    return {"cte": ctx_dict, "tkg": tkg_list}


def visualize_tensor_differences_over_steps(
    dir_cpu,
    dir_neuron,
    include_ctx=True,
    max_elements=4096,
):
    import os
    dir_cpu = os.path.expanduser(dir_cpu)
    dir_neuron = os.path.expanduser(dir_neuron)

    cpu_res = load_ctx_tkg_from_dir(dir_cpu)
    neu_res = load_ctx_tkg_from_dir(dir_neuron)

    # CTX
    if include_ctx and cpu_res.get("cte") and neu_res.get("cte"):
        plot_difference_visualizations_from_flatdicts(
            cpu_res["cte"], neu_res["cte"],
            max_elements=max_elements,
            tkg_boost=False,
            phase="CTE",
            step=None,
        )

    # TKG steps
    cpu_steps = cpu_res.get("tkg", []) or []
    neu_steps = neu_res.get("tkg", []) or []
    for i in range(min(len(cpu_steps), len(neu_steps))):
        plot_difference_visualizations_from_flatdicts(
            cpu_steps[i], neu_steps[i],
            max_elements=max_elements,
            tkg_boost=True,           # larger markers for tiny TKG tensors
            phase="TKG",
            step=i + 1,
        )
        