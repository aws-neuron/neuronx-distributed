from functools import partial
from typing import Any, Callable


class PostPartitionHooks:
    def __init__(
        self,
    ):
        self.hooks = []

    def register_post_partition_hook(self, callable_function: Callable[..., Any], func_args=(), func_kwargs={}):
        if not callable(callable_function):
            raise ValueError("callable_function must be a callable object")

        self.hooks.append(
            {
                "function": partial(callable_function, *func_args, **func_kwargs),
                "name": callable_function.__name__,
            }
        )

    def execute_all_hooks(self, model=None):
        hook_outputs = []
        for hook in self.hooks:
            func = hook["function"]
            name = hook["name"]
            if name == "filter_to_local_parameter_group":
                assert model is not None, "When executing filter_to_local_parameter_group hook, model object cannot be None"
                hook_outputs.append(func(model=model))
            else:
                hook_outputs.append(func())

        # Finally, clear all hooks
        self.hooks.clear()
        return hook_outputs
