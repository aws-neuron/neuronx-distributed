import torch


class NxDModel(torch.nn.Module):
    def __init__(self, module, nxd_config):
        super().__init__()

        self.module = module
        self.nxd_config = nxd_config

        self.pp_enabled = nxd_config["pipeline_parallel_size"] > 1

        self.train()

    def __repr__(self):
        return "NxDModel({})".format(self.module.__repr__())

    def local_module(self):
        if self.pp_enabled:
            return self.module.local_module
        return self.module

    def original_module(self):
        if self.pp_enabled:
            return self.module.original_torch_module
        return self.module

    def run_train(self, *args, **kwargs):
        if self.pp_enabled:
            return self.module.run_train(*args, **kwargs)
        loss = self.forward(*args, **kwargs)
        loss.backward()
        return loss

    def run_eval(self, *args, **kwargs):
        assert self.pp_enabled, "`run_eval` should be used only when pipeline parallel is enabled."
        return self.module.run_eval(*args, **kwargs)

    """
    torch.nn.Module APIs
    """

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        if self.pp_enabled:
            for n, p in self.module.local_named_parameters(*args, **kwargs):
                yield n, p
            return
        for n, p in self.module.named_parameters(*args, **kwargs):
            yield n, p

    def named_buffers(self, *args, **kwargs):
        if self.pp_enabled:
            for n, b in self.module.local_named_buffers(*args, **kwargs):
                yield n, b
            return
        for n, b in self.module.named_buffers(*args, **kwargs):
            yield n, b

    def named_children(self):
        if self.pp_enabled:
            for n, c in self.module.local_named_children():
                yield n, c
            return
        for n, c in self.module.named_children():
            yield n, c

    def named_modules(self, *args, **kwargs):
        if self.pp_enabled:
            for n, m in self.module.local_named_modules(*args, **kwargs):
                yield n, m
            return
        for n, m in self.module.named_modules(*args, **kwargs):
            yield n, m

    def state_dict(self, *args, **kwargs):
        if self.pp_enabled:
            return self.module.local_state_dict(*args, **kwargs)
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)

    """
    common transformers.PreTrainedModel APIs
    """

    @property
    def dtype(self):
        return self.original_module().dtype

    @property
    def config(self):
        return self.original_module().config

    @property
    def name_or_path(self):
        return self.original_module().name_or_path
