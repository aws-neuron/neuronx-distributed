import os
from multiprocessing.queues import SimpleQueue
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch.multiprocessing as mp
from lightning_fabric.strategies.launchers.xla import _rank_teardown
from pytorch_lightning.strategies.launchers.multiprocessing import _GlobalStateSnapshot
from pytorch_lightning.strategies.launchers.xla import _XLALauncher


class _NeuronXLALauncher(_XLALauncher):
    r"""Launches processes that run a given function in parallel on XLA supported hardware, and joins them all at the
    end.

    The main process in which this launcher is invoked creates N so-called worker processes (using the
    `torch_xla` :func:`xmp.spawn`) that run the given function.
    Worker processes have a rank that ranges from 0 to N - 1.

    Note:
        - This launcher requires all objects to be pickleable.
        - It is important that the entry point to the program/script is guarded by ``if __name__ == "__main__"``.

    Args:
        strategy: A reference to the strategy that is used together with this launcher

    """

    def launch(self, function: Callable, *args: Any, trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        """Launches processes that run the given function in parallel.

        The function is allowed to have a return value. However, when all processes join, only the return value
        of worker process 0 gets returned from this `launch` method in the main process.

        Arguments:
            function: The entry point for all launched processes.
            *args: Optional positional arguments to be passed to the given function.
            trainer: Optional reference to the :class:`~lightning.pytorch.trainer.trainer.Trainer` for which
                a selected set of attributes get restored in the main process after processes join.
            **kwargs: Optional keyword arguments to be passed to the given function.

        """
        if not self._strategy.cluster_environment.creates_processes_externally:
            context = mp.get_context(self._start_method)
            return_queue = context.SimpleQueue()
            import torch_xla.distributed.xla_multiprocessing as xmp

            process_context = xmp.spawn(
                self._wrapping_function,
                args=(trainer, function, args, kwargs, return_queue),
                nprocs=self._strategy.num_processes,
                start_method=self._start_method,
                join=False,  # we will join ourselves to get the process references
            )
            # xla will not actually create processes if only 1 device
            if process_context is not None:
                self.procs = process_context.processes
                while not process_context.join():
                    pass

            worker_output = return_queue.get()
            if trainer is None:
                return worker_output

            self._recover_results_in_main_process(worker_output, trainer)
            return worker_output.trainer_results
        else:  # Neuron change for launch with torchrun
            process_idx = int(os.environ["LOCAL_RANK"])
            self._strategy._local_rank = process_idx
            results = function(*args, **kwargs)
            _rank_teardown(process_idx)
            return results

    def _wrapping_function(
        self,
        # XLA's multiprocessing returns the global index, not the local index as torch's multiprocessing
        # https://github.com/pytorch/xla/blob/v1.13.0/torch_xla/distributed/xla_multiprocessing.py#L321
        process_idx: int,
        trainer: Optional["pl.Trainer"],
        function: Callable,
        args: Any,
        kwargs: Any,
        return_queue: SimpleQueue,
        global_states: Optional[_GlobalStateSnapshot] = None,
    ) -> None:
        results = function(*args, **kwargs)

        if trainer is not None:
            results = self._collect_rank_zero_results(trainer, results)

        _rank_teardown(self._strategy.local_rank)
