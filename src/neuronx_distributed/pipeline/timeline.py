from neuronx_distributed.parallel_layers.parallel_state import (
    gather_python_object,
    get_data_parallel_rank,
    get_tensor_model_parallel_rank,
)
from neuronx_distributed.pipeline.comm import get_gloo_pg_for_first_pp_group
from neuronx_distributed.utils.timeline import Timeline


class PPTimeline(Timeline):
    def __init__(self, trace_file_path, pp_rank):
        super().__init__(trace_file_path, pp_rank)
        if self.enabled:
            self.group = get_gloo_pg_for_first_pp_group()

    @property
    def should_record(self):
        return self.enabled and get_data_parallel_rank() == 0 and get_tensor_model_parallel_rank() == 0

    def _collect_events_for_all_ranks(self):
        self.all_rank_events = gather_python_object(self.current_rank_events, group=self.group)
