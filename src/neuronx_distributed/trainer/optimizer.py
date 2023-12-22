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
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        self.optimizer.__setstate__(state)

    def step(self, closure=None):
        # sequence parallel all-reduce
        if self.nxd_config["sequence_parallel"]:
            grads.allreduce_sequence_parallel_gradients(self)

        optimizer_config = self.nxd_config["optimizer_config"]
        if not optimizer_config["zero_one_enabled"]:
            grads.bucket_allreduce_gradients(xm._fetch_gradients(self))
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
