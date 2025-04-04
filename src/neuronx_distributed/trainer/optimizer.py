import torch
import torch_xla.core.xla_model as xm

from neuronx_distributed.parallel_layers import grads
from neuronx_distributed.utils.logger import get_logger

logger = get_logger()


class NxDOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, nxd_config):
        self.optimizer = optimizer
        self.nxd_config = nxd_config

        self._grad_norm = None

        # fetch parameters
        self.params = []
        for param_group in self.param_groups:
            for param in param_group["params"]:
                self.params.append(param)

    def __repr__(self):
        return "NxDOptimizer({})".format(self.optimizer.__repr__())

    @property
    def grad_norm(self):
        return self._grad_norm

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, state):
        self.optimizer.state = state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self.optimizer.param_groups = param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optimizer.defaults = defaults

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict = self._mark_expert_parallel_states(state_dict)
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        self.optimizer.__setstate__(state)


    def _mark_expert_parallel_states(self, state_dict):
        if state_dict is None:
            return None

        ep_ids = set()
        idx = 0
        param_set = set()
        for param_group in self.__getstate__()["param_groups"]:
            for group, params in param_group.items():
                if group == "params":
                    for p in params:
                        if isinstance(p, torch.Tensor) and hasattr(p, "expert_model_parallel") and p.expert_model_parallel:
                            if id(p) not in param_set:
                                ep_ids.add(idx)
                        idx += 1
                        param_set.add(id(p))


        for id_p, param_state_dict in state_dict["state"].items():
            if id_p in ep_ids:
                for state_key in param_state_dict:
                    param_state_dict[state_key].expert_model_parallel = True

        return state_dict


    def _fetch_gradients(self):
        gradients = []
        ep_gradients = []
        for param_group in self.optimizer.__getstate__()["param_groups"]:
            for group, params in param_group.items():
                if group == "params":
                    for p in params:
                        if isinstance(p, torch.Tensor):
                            if p.grad is not None:
                                if hasattr(p, "expert_model_parallel") and p.expert_model_parallel:
                                    ep_gradients.append(p.grad.data)
                                else:
                                    gradients.append(p.grad.data)
                            elif hasattr(p, "main_grad"):
                                if hasattr(p, "expert_model_parallel") and p.expert_model_parallel:
                                    ep_gradients.append(p.main_grad.data)
                                else:
                                    gradients.append(p.main_grad.data)

        return gradients, ep_gradients

    def step(self, closure=None):
        # context parallel all-reduce
        grads.allreduce_context_parallel_gradients(self)
        
        # sequence parallel all-reduce
        if self.nxd_config["sequence_parallel"]:
            grads.allreduce_sequence_parallel_gradients(self)

        optimizer_config = self.nxd_config["optimizer_config"]
        if not optimizer_config["zero_one_enabled"]:
            non_ep_gradients, ep_gradients = self._fetch_gradients()
            grads.bucket_allreduce_gradients(non_ep_gradients + ep_gradients)
            if len(ep_gradients) > 0:
                # initial allreduce takes place over the expert data parallel group
                # which coincides with data parallel group when ep is disabled. when ep
                # is enabled, non-ep gradients would additionally need to be reduced
                # over the expert model parallel groups (since ep happens over dp ranks).
                # non-ep gradient reduction needs to take place separately over emp/edp
                # groups (in two separate steps) to side-step the MPMD limitation in runtime.
                grads.bucket_allreduce_gradients(non_ep_gradients, reduce_over_ep_group=True)
            if optimizer_config["grad_clipping"]:
                self._grad_norm = grads.clip_grad_norm(self.params, optimizer_config["max_grad_norm"])
        ret = self.optimizer.step(closure=closure)
        if optimizer_config["zero_one_enabled"]:
            self._grad_norm = self.optimizer.grad_norm
        return ret

    # [TODO] Remove this method
    def save_state_dict(self, output_dir, num_workers_per_step=8):
        logger.info("`NxDOptimizer.save_state_dict` is deprecated, please use `nxd.save_checkpoint` instead.")

        optimizer_config = self.nxd_config["optimizer_config"]
        assert optimizer_config["zero_one_enabled"]
        self.optimizer.save_sharded_state_dict(output_dir, num_workers_per_step)

    # [TODO] Remove this method
    def load_state_dict_from(self, output_dir, num_workers_per_step=8):
        logger.info("`NxDOptimizer.load_state_dict_from` is deprecated, please use `nxd.load_checkpoint` instead.")

        optimizer_config = self.nxd_config["optimizer_config"]
        assert optimizer_config["zero_one_enabled"]
        self.optimizer.load_sharded_state_dict(output_dir, num_workers_per_step)
