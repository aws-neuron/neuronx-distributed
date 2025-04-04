from contextlib import contextmanager
from typing import Iterator, Any
import torch.distributed
from torch.distributed import ProcessGroup
from unittest.mock import MagicMock, patch


class MockDistributed(MagicMock):

    _backend = torch.distributed
    _rank = 0
    _world_size = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MockDistributed._world_size = kwargs.get("world_size", 1)

    @staticmethod
    def is_initialized():
        return True

    @staticmethod
    def init_process_group(backend, rank=0, world_size=1):
        MockDistributed._rank = rank
        MockDistributed._world_size = world_size

    @staticmethod
    def get_rank(group=None):
        return MockDistributed._rank

    @staticmethod
    def get_backend(group=None):
        return "xla"
    
    @staticmethod
    def get_world_size(group=None):
        if group is not None:
            return group.size()
        return MockDistributed._world_size

    @staticmethod
    def destroy_process_group(group=None):
        pass

    @staticmethod
    def new_group(group_ranks, pg_options):
        mesh = pg_options["xla_pg_options"]["mesh"]
        mock_pg = MagicMock(spec=ProcessGroup)
        mock_pg.group_ranks = group_ranks
        mock_pg._mesh = mesh
        mock_pg.rank.return_value = 0
        mock_pg.size.return_value = len(mesh[0])
        return mock_pg # type: ignore

    def __getattribute__(self, name):
        try:
            # invoke overridden method
            return object.__getattribute__(self, name)
        except AttributeError:
            # if not found, invoke from torch.distributed
            return getattr(MockDistributed._backend, name)


@contextmanager
def mock_distributed(world_size: int) -> Iterator[Any]:
    """
    Mock torch.distributed backend

    This function is mainly used for mocking torch distributed module for tracing purpose, where
    NxD only generated HLO for rank-0.

    Usage:
        >>> with mock_distributed(world_size=world_size):
        >>>     # init distributed backend and nxd
        >>>     torch.distributed.init_process_group(backend="xla", rank, world_size)
        >>>     nxd.parallel_layers.parallel_state.initialize_model_parallel(...)
        >>>     ...
        >>>     # cleanup
        >>>     parallel_state.destroy_model_parallel()
        >>>     torch.distributed.destroy_process_group()
    """
    with patch('torch.distributed', MockDistributed(world_size=world_size)) as m1, \
        patch("torch_xla.core.xla_model.xrt_world_size", return_value=world_size) as m2, \
        patch('torch_xla.runtime.world_size', return_value=world_size) as m3:
        yield (m1, m2, m3)
