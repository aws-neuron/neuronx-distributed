from typing import List, Optional
import torch
from torch.distributed import ProcessGroup

from neuronx_distributed.parallel_layers.parallel_state import initialize_model_parallel

_MOE_TKG_TENSOR_MODEL_PARALLEL_GROUP: Optional[ProcessGroup] = None
_MOE_TKG_EXPERT_MODEL_PARALLEL_GROUP: Optional[ProcessGroup] = None
_MOE_CTE_TENSOR_MODEL_PARALLEL_GROUP: Optional[ProcessGroup] = None
_MOE_CTE_EXPERT_MODEL_PARALLEL_GROUP: Optional[ProcessGroup] = None

def init_tensor_expert_parallel_moe_process_groups(tkg_tp_degree: int, tkg_ep_degree: int, cte_tp_degree: int, cte_ep_degree: int):
    global _MOE_TKG_TENSOR_MODEL_PARALLEL_GROUP
    global _MOE_TKG_EXPERT_MODEL_PARALLEL_GROUP
    global _MOE_CTE_TENSOR_MODEL_PARALLEL_GROUP
    global _MOE_CTE_EXPERT_MODEL_PARALLEL_GROUP

    # TODO: This should use get_platform_lnc, however since we are doing the process group creation before MoE initialization currently, the preferred approach can not be done
    lnc_size = 1 if cte_tp_degree * cte_ep_degree <= 32 else 2

    if _MOE_TKG_EXPERT_MODEL_PARALLEL_GROUP is None and _MOE_TKG_TENSOR_MODEL_PARALLEL_GROUP is None:
        moe_replica_groups = initialize_model_parallel(tensor_model_parallel_size = tkg_tp_degree, expert_model_parallel_size = tkg_ep_degree, mesh_only = True, lnc_size=lnc_size)
        tkg_ep_tp_group_mesh = moe_replica_groups.tp_groups
        tkg_ep_tp_group = torch.distributed.new_group(
            tkg_ep_tp_group_mesh[0], pg_options={"xla_pg_options": {"mesh": tkg_ep_tp_group_mesh}}
        )
        _MOE_TKG_TENSOR_MODEL_PARALLEL_GROUP = tkg_ep_tp_group

        tkg_ep_group_mesh = moe_replica_groups.ep_model_groups
        tkg_ep_group = torch.distributed.new_group(
            tkg_ep_group_mesh[0], pg_options={"xla_pg_options": {"mesh": tkg_ep_group_mesh}}
        )
        _MOE_TKG_EXPERT_MODEL_PARALLEL_GROUP = tkg_ep_group

    if _MOE_CTE_EXPERT_MODEL_PARALLEL_GROUP is None and _MOE_CTE_TENSOR_MODEL_PARALLEL_GROUP is None:
        moe_replica_groups = initialize_model_parallel(tensor_model_parallel_size = cte_tp_degree, expert_model_parallel_size = cte_ep_degree, mesh_only = True, lnc_size=lnc_size)
        cte_ep_tp_group_mesh = moe_replica_groups.tp_groups
        cte_ep_tp_group = torch.distributed.new_group(
            cte_ep_tp_group_mesh[0], pg_options={"xla_pg_options": {"mesh": cte_ep_tp_group_mesh}}
        )
        _MOE_CTE_TENSOR_MODEL_PARALLEL_GROUP = cte_ep_tp_group

        cte_ep_group_mesh = moe_replica_groups.ep_model_groups
        cte_ep_group = torch.distributed.new_group(
            cte_ep_group_mesh[0], pg_options={"xla_pg_options": {"mesh": cte_ep_group_mesh}}
        )
        _MOE_CTE_EXPERT_MODEL_PARALLEL_GROUP = cte_ep_group

def get_moe_tp_ep_group(prefill: bool = True):
    if prefill:
        assert _MOE_CTE_TENSOR_MODEL_PARALLEL_GROUP is not None, "_MOE_CTE_TENSOR_MODEL_PARALLEL_GROUP is not initialized"
        return _MOE_CTE_TENSOR_MODEL_PARALLEL_GROUP
    else:
        assert _MOE_TKG_TENSOR_MODEL_PARALLEL_GROUP is not None, "_MOE_TKG_TENSOR_MODEL_PARALLEL_GROUP is not initialized"
        return _MOE_TKG_TENSOR_MODEL_PARALLEL_GROUP

def get_moe_ep_group(prefill: bool = True):
    if prefill:
        assert _MOE_CTE_EXPERT_MODEL_PARALLEL_GROUP is not None, "_MOE_CTE_EXPERT_MODEL_PARALLEL_GROUP is not initialized"
        return _MOE_CTE_EXPERT_MODEL_PARALLEL_GROUP
    else:
        assert _MOE_TKG_EXPERT_MODEL_PARALLEL_GROUP is not None, "_MOE_TKG_EXPERT_MODEL_PARALLEL_GROUP is not initialized"
        return _MOE_TKG_EXPERT_MODEL_PARALLEL_GROUP