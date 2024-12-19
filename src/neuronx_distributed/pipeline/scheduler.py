from abc import ABC, abstractmethod
from typing import List, Union

class PipelineTask:
    def __init__(self, mb, model_chunk=0, graph_break=True):
        self.mb = mb
        self.model_chunk = model_chunk
        self.graph_break = graph_break

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self.mb == other.mb
            and self.model_chunk == other.model_chunk
            and self.graph_break == other.graph_break
        )


class ForwardStepTask(PipelineTask):
    def __repr__(self):
        return f"ForwardStepTask_microbatch_{self.mb}_modelchunk_{self.model_chunk}_graphbreak_{self.graph_break}"


class ForwardPreprocessTask(PipelineTask):
    def __repr__(self):
        return f"ForwardPreprocessTask_microbatch_{self.mb}_modelchunk_{self.model_chunk}_graphbreak_{self.graph_break}"


class ForwardPostprocessTask(PipelineTask):
    def __repr__(self):
        return (
            f"ForwardPostprocessTask_microbatch_{self.mb}_modelchunk_{self.model_chunk}_graphbreak_{self.graph_break}"
        )


class BackwardStepTask(PipelineTask):
    def __repr__(self):
        return f"BackwardStepTask_microbatch_{self.mb}_modelchunk_{self.model_chunk}_graphbreak_{self.graph_break}"


class BackwardPreprocessTask(PipelineTask):
    def __repr__(self):
        return (
            f"BackwardPreprocessTask_microbatch_{self.mb}_modelchunk_{self.model_chunk}_graphbreak_{self.graph_break}"
        )


class BackwardPostprocessTask(PipelineTask):
    def __repr__(self):
        return (
            f"BackwardPostprocessTask_microbatch_{self.mb}_modelchunk_{self.model_chunk}_graphbreak_{self.graph_break}"
        )


class PostProcessTask:
    def __init__(self, graph_break=True):
        """
        PostProcessTask happens after pipeline execution
        """
        self.mb = -1
        self.model_chunk = -1
        self.graph_break = graph_break


class ReduceGradsTask(PostProcessTask):
    def __repr__(self):
        return "ReduceGradsTask"

    def __eq__(self, other) -> bool:
        return type(self) is type(other)


class PipeSchedule(ABC):
    """Directs the execution of a pipeline engine by generating sequences of
    :class:`PipelineTask`.

    Schedules are generators that yield sequences of
    :class:`PipelineTask` to process the micro-batches in one batch.
    Each yielded step is atomic in the sense that a barrier
    synchronization can be placed between successive steps without
    deadlock.

    Args:
        num_microbatches (int): The number of micro-batches that comprise a batch.
        stages (int): The number of pipeline stages.
        stage_id (int): The pipe stage that will execute the generated schedule.
    """

    def __init__(self, num_microbatches, stages, stage_id):
        super().__init__()
        self.num_microbatches = num_microbatches
        self.stages = stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

    @abstractmethod
    def steps(self):
        """Yield a list of :class:`PipelineTask` for each step in the schedule.

        .. note::
            Schedules must implement ``steps()`` to define the schedule.

        Returns:
            Instructions to be executed as one step of the pipeline
        """

    def _valid_micro_batch(self, micro_batch_id):
        return 0 <= micro_batch_id < self.num_microbatches

    def _valid_stage(self, stage_id):
        return 0 <= stage_id < self.stages

    @property
    def stage(self):
        """Stage index used to configure this schedule."""
        return self.stage_id

    @property
    def num_stages(self):
        """The number of total pipeline stages used to configure this schedule."""
        return self.stages

    @property
    def is_first_stage(self):
        """True if the configured ``stage_id`` is the first stage in the pipeline."""
        return self.stage_id == 0

    @property
    def is_last_stage(self):
        """True if the configured ``stage_id`` is the last stage in the pipeline."""
        return self.stage_id == self.stages - 1

    def __iter__(self):
        self.it = None
        return self

    def __next__(self):
        if self.it is None:
            self.it = self.steps()
        return next(self.it)


class InferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism."""

    def steps(self):
        total_steps = self.num_microbatches
        for micro_batch_id in range(total_steps):
            cmds: List[PipelineTask] = []
            cmds.append(ForwardPreprocessTask(micro_batch_id))
            cmds.append(ForwardStepTask(micro_batch_id))
            cmds.append(ForwardPostprocessTask(micro_batch_id))
            yield cmds


class Train1F1BSchedule(PipeSchedule):
    """
    Refactor the TrainSchedule to be microbatch based schedule,
    which will be more aligned with the interleaved scheduler

    1F1B non-interleaved scheduler, below is an example with PP4 and 6 microbatches
          |-Warmup-|             |-------steady--------|      |--remaining--|     # noqa
    PP0   F0  F1  F2              F3  B0  F4  B1  F5  B2      B3     B4    B5
    PP1       F0  F1          F2  B0  F3  B1  F4  B2  F5  B3      B4    B5
    PP2           F0      F1  B0  F2  B1  F3  B2  F4  B3  F5  B4     B5
    PP3               F0  B0  F1  B1  F2  B2  F3  B3  F4  B4  F5  B5

    There are three states
    - Warmup state, forward only
    - Steady state, 1 forward and 1 backward
    - Remaining state, backward only
    """

    def __init__(self, num_microbatches, stages, stage_id):
        super().__init__(num_microbatches, stages, stage_id)
        self.get_microbatche_schedule()

    def get_microbatche_schedule(self):
        num_warmup_steps: int = self.stages - self.stage_id - 1
        num_warmup_steps = min(num_warmup_steps, self.num_microbatches)
        self.num_warmup_steps = num_warmup_steps
        self.num_steady_state_microbatches = self.num_microbatches - num_warmup_steps
        self.num_remaining_microbatches = num_warmup_steps

    def _step_to_micro_batch(self, step_id):
        """
        Given a step id, return the corresponding microbatch id and whether current step is doing forward
        """
        # Warmup phase
        if step_id < self.num_warmup_steps:
            return step_id, True
        # Steady 1F1B phase
        elif step_id < self.num_warmup_steps + self.num_steady_state_microbatches * 2:
            current_1f1b_step = step_id - self.num_warmup_steps
            is_forward = current_1f1b_step % 2 == 0
            if is_forward:
                current_mb = current_1f1b_step // 2 + self.num_warmup_steps
            else:
                current_mb = current_1f1b_step // 2
            return current_mb, is_forward
        # Cool down phase
        else:
            current_remaining_step = step_id - self.num_warmup_steps - self.num_steady_state_microbatches * 2
            current_mb = self.num_steady_state_microbatches + current_remaining_step
            return current_mb, False

    def steps(self):
        total_steps = 2 * self.num_microbatches + 1
        prev_micro_batch_id = -1
        for step_id in range(total_steps):
            # Last step for grad reduction
            cmds: List[Union[PipelineTask, ReduceGradsTask]] = []
            if step_id == total_steps - 1:
                cmds.append(ReduceGradsTask())
                yield cmds
                return

            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            # Preprocess
            if is_forward:
                # Always do ForwardPreprocessTask, which only recv fwd when PP rank > 0
                cmds.append(ForwardPreprocessTask(micro_batch_id))
            else:
                # Once enter the 1F1B steady states
                # Need to recv bwd before sending fwd to avoid deadlock
                # Recv bwd for non-last-stage
                if self._valid_stage(self.next_stage):
                    cmds.append(BackwardPreprocessTask(micro_batch_id))
                # Send fwd during steady state
                if micro_batch_id < self.num_steady_state_microbatches and self._valid_stage(self.next_stage):
                    cmds.append(ForwardPostprocessTask(prev_micro_batch_id))

            # Computation
            if is_forward:
                cmds.append(ForwardStepTask(micro_batch_id))
            else:
                cmds.append(BackwardStepTask(micro_batch_id))

            # Postprocess
            if not is_forward:
                # Send bwd
                if self._valid_stage(self.prev_stage):
                    cmds.append(BackwardPostprocessTask(micro_batch_id))
            else:
                # Only send fwd during warmup, steady states this send is handled by bwd mbs
                if micro_batch_id < self.num_warmup_steps and self._valid_stage(self.next_stage):
                    cmds.append(ForwardPostprocessTask(micro_batch_id))

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds


class TrainInterleavedSchedule(PipeSchedule):
    """
    Interleaved pipelining schedule adopted from Megatron/Apex
    https://github.com/NVIDIA/apex/blob/master/apex/transformer/pipeline_parallel/schedules/fwd_bwd_pipelining_with_interleaving.py
    Communication general rules
        - Recv forward: recvs from the previous stage ---> recv_prev
        - Send forward: sends to the next stage ---> send_next
        - Recv backward: recvs from next stage ---> recv_next
        - Send backward: send to previous stage ---> send_prev
    Priciples
        - Send is always for the current step microbatch/model_chunk
        - Recv is always for the next step microbatch/model_chunk
    Step:
        A step can contain different components in different pipeline phases
            - Warmup phase: A step contains a single mb forward for a model chunk
            - Steady states: A step contains a forward and a backward, can be different mb and model chunk
            - Cool down phase: A step contains a single mb backward for a model chunk
    """

    def __init__(
        self,
        num_microbatches,
        num_model_chunks,
        stages,
        stage_id,
        fused_send_recv=False,
        fused_fwd_bwd=False,
        use_odd_even_scheduler=False,
    ):
        super().__init__(num_microbatches, stages, stage_id)
        # We do not need to fuse graph when there is no steady states
        if num_microbatches <= stages:
            fused_send_recv = False
            fused_fwd_bwd = False
        self.num_model_chunks = num_model_chunks
        self.fused_send_recv = fused_send_recv
        self.fused_fwd_bwd = fused_fwd_bwd
        self.use_odd_even_scheduler = use_odd_even_scheduler
        self.get_step_schedule()

    def get_step_schedule(self):
        """
        Run all forward passes and then all backward passes if number of
        microbatches is just the number of pipeline stages.
        Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        all workers, followed by more microbatches after depending on
        stage ID (more forward passes for earlier stages, later stages can
        immediately start with 1F1B).
        """
        if self.num_microbatches % self.stages != 0:
            raise ValueError(
                f"Interleaved pipeline requires num_microbatches % pipeline_parallel_size == 0, current num_microbatches {self.num_microbatches} and pipeline_parallel_size {self.stages}"
            )
        self.num_microbatches_steps: int = self.num_microbatches * self.num_model_chunks
        if self.num_microbatches == self.stages:
            self.num_warmup_steps = self.num_microbatches_steps
        else:
            num_warmup_steps = (self.stages - self.stage_id - 1) * 2
            num_warmup_steps += (self.num_model_chunks - 1) * self.stages
            self.num_warmup_steps = min(num_warmup_steps, self.num_microbatches_steps)
        self.num_steady_state_steps: int = self.num_microbatches_steps - self.num_warmup_steps
        self.num_remaining_steps: int = self.num_warmup_steps

    def get_model_chunk_id(self, step_id, is_forward=True):
        """
        Helper function to get the model chunk ID given the iteration number.

        Each model chunk processes pipeline_parallel_size microbatches
        at a time. We assume that the number of microbatches is a
        multiple of pipeline_parallel_size*num_model_chunks.
        """
        # backward is late compared with fwd for num_warmup_steps
        if not is_forward:
            step_id -= self.num_warmup_steps
        microbatch_group_size = self.stages * self.num_model_chunks
        microbatch_id_in_group = step_id % microbatch_group_size
        model_chunk_id = microbatch_id_in_group // self.stages
        if not is_forward:
            model_chunk_id = self.num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id(self, step_id, is_forward=True):
        """
        Helper function to get the microbatch ID given the iteration number.

        Each microbatch group contains PP_size microbatches for all model chunks.
        We first decide which microbatch group the current step belongs, then find the microbatch index
        in the microbatch group.
        """
        # backward is late compared with fwd for num_warmup_steps
        if not is_forward:
            step_id -= self.num_warmup_steps
        microbatch_group_size = self.stages * self.num_model_chunks
        microbatch_group_id = step_id // microbatch_group_size
        microbatch_id_in_group = step_id % microbatch_group_size
        passed_microbatches = self.stages * microbatch_group_id
        microbatch_idx_current_group = microbatch_id_in_group % self.stages
        return passed_microbatches + microbatch_idx_current_group

    def _should_recv_fwd_send_bwd(self, step_id, is_forward=True):
        """
        recv_prev: fwd started from first stage first model chunk
        send_prev: bwd stopped at first stage first model chunk
        recv_prev/send_prev = next_mb/mb not is_first_stage_and_first_model_chunk
        """
        # recv_prev for next step forward
        if is_forward:
            step_id += 1
        # send_next for current step backward
        model_chunk = self.get_model_chunk_id(step_id, is_forward=is_forward)
        return not (self.stage_id == 0 and model_chunk == 0)

    def _should_recv_bwd_send_fwd(self, step_id, is_forward=True):
        """
        recv_next: bwd started from last stage last model chunk
        send_next: fwd stopped at last stage last model chunk
        recv_next/send_next = next_mb/mb not is_last_stage_and_last_model_chunk
        """
        # recv_next for next step backward
        if not is_forward:
            step_id += 1
        # send_next for current step forward
        model_chunk = self.get_model_chunk_id(step_id, is_forward=is_forward)
        return not (self.stage_id == self.stages - 1 and model_chunk == self.num_model_chunks - 1)

    def _get_forward_preprocess_task(self, step_id, graph_break=True):
        """
        Recv forward, this is for next step
        """
        next_mb_fwd = self.get_microbatch_id(step_id + 1, is_forward=True)
        next_model_chunk_fwd = self.get_model_chunk_id(step_id + 1, is_forward=True)
        return ForwardPreprocessTask(next_mb_fwd, model_chunk=next_model_chunk_fwd, graph_break=graph_break)

    def _get_forward_postprocess_task(self, step_id, graph_break=True):
        """
        Send forward, this is for current step
        """
        current_mb_fwd = self.get_microbatch_id(step_id, is_forward=True)
        current_model_chunk_fwd = self.get_model_chunk_id(step_id, is_forward=True)
        return ForwardPostprocessTask(current_mb_fwd, model_chunk=current_model_chunk_fwd, graph_break=graph_break)

    def _get_backward_preprocess_task(self, step_id, graph_break=True):
        """
        Recv backward, this is for next step
        """
        next_mb_bwd = self.get_microbatch_id(step_id + 1, is_forward=False)
        next_model_chunk_bwd = self.get_model_chunk_id(step_id + 1, is_forward=False)
        return BackwardPreprocessTask(next_mb_bwd, model_chunk=next_model_chunk_bwd, graph_break=graph_break)

    def _get_backward_postprocess_task(self, step_id, graph_break=True):
        """
        Send backward, this is for current step
        """
        current_mb_bwd = self.get_microbatch_id(step_id, is_forward=False)
        current_model_chunk_bwd = self.get_model_chunk_id(step_id, is_forward=False)
        return BackwardPostprocessTask(current_mb_bwd, model_chunk=current_model_chunk_bwd, graph_break=graph_break)

    def _get_forward_step_task(self, step_id, graph_break=True):
        current_mb_bwd = self.get_microbatch_id(step_id, is_forward=True)
        current_model_chunk_fwd = self.get_model_chunk_id(step_id, is_forward=True)
        return ForwardStepTask(current_mb_bwd, model_chunk=current_model_chunk_fwd, graph_break=graph_break)

    def _get_backward_step_task(self, step_id, graph_break=True):
        current_mb_bwd = self.get_microbatch_id(step_id, is_forward=False)
        current_model_chunk_bwd = self.get_model_chunk_id(step_id, is_forward=False)
        return BackwardStepTask(current_mb_bwd, model_chunk=current_model_chunk_bwd, graph_break=graph_break)

    def _add_pre_post_processing_tasks(self, step_id, cmds, fwd_pre=True, fwd_post=True, bwd_pre=True, bwd_post=True):
        """
        send_fwd, recv_fwd, send_bwd, recv_bwd
        Specialy handling for last stage to send first before recv to remove deadlocks
        """
        if not self.use_odd_even_scheduler:
            if not self.stage_id == self.num_stages - 1:
                # recv fwd
                if fwd_pre:
                    cmds.append(
                        self._get_forward_preprocess_task(step_id, graph_break=not self.fused_send_recv or not bwd_post)
                    )
                # send bwd
                if bwd_post:
                    cmds.append(self._get_backward_postprocess_task(step_id))
                # send fwd
                if fwd_post:
                    cmds.append(
                        self._get_forward_postprocess_task(step_id, graph_break=not self.fused_send_recv or not bwd_pre)
                    )
                # recv bwd
                if bwd_pre:
                    cmds.append(self._get_backward_preprocess_task(step_id))
            else:
                # send_fwd
                if fwd_post:
                    cmds.append(
                        self._get_forward_postprocess_task(step_id, graph_break=not self.fused_send_recv or not bwd_pre)
                    )
                # recv_bwd
                if bwd_pre:
                    cmds.append(self._get_backward_preprocess_task(step_id))
                # recv_fwd
                if fwd_pre:
                    cmds.append(
                        self._get_forward_preprocess_task(step_id, graph_break=not self.fused_send_recv or not bwd_post)
                    )
                # send_bwd
                if bwd_post:
                    cmds.append(self._get_backward_postprocess_task(step_id))
        else:
            if self.stage_id % 2 == 0:  # Schedule for Even stages
                # recv fwd
                if fwd_pre:
                    cmds.append(self._get_forward_preprocess_task(step_id))
                # send fwd
                if fwd_post:
                    cmds.append(self._get_forward_postprocess_task(step_id))
                # send bwd
                if bwd_post:
                    cmds.append(self._get_backward_postprocess_task(step_id))
                # recv bwd
                if bwd_pre:
                    cmds.append(self._get_backward_preprocess_task(step_id))
            else:  # Schedule for Odd stages
                # send_fwd
                if fwd_post:
                    cmds.append(self._get_forward_postprocess_task(step_id))
                # recv_fwd
                if fwd_pre:
                    cmds.append(self._get_forward_preprocess_task(step_id))
                # recv_bwd
                if bwd_pre:
                    cmds.append(self._get_backward_preprocess_task(step_id))
                # send_bwd
                if bwd_post:
                    cmds.append(self._get_backward_postprocess_task(step_id))

    def steps(self):
        total_steps = self.num_warmup_steps + self.num_steady_state_steps + self.num_remaining_steps + 1
        for step_id in range(total_steps):
            cmds = []
            if step_id == total_steps - 1:
                cmds.append(ReduceGradsTask())
                yield cmds
                return

            # Warmup
            if step_id < self.num_warmup_steps:
                # Recv forward for the first mb
                if step_id == 0:
                    cmds.append(self._get_forward_preprocess_task(-1))
                cmds.append(self._get_forward_step_task(step_id))
                # Do not recv fwd if it is all warmup batches and we have reached the last warmup batch
                recv_fwd = step_id != (self.num_microbatches_steps - 1)
                send_fwd = self._should_recv_bwd_send_fwd(step_id, is_forward=True)
                # Only recv bwd when we are at the last warm up batch
                recv_bwd = step_id == self.num_warmup_steps - 1 and self._should_recv_bwd_send_fwd(
                    step_id, is_forward=False
                )
                send_bwd = False
                self._add_pre_post_processing_tasks(
                    step_id, cmds, fwd_pre=recv_fwd, fwd_post=send_fwd, bwd_pre=recv_bwd, bwd_post=send_bwd
                )
            # Steady state
            elif step_id < self.num_warmup_steps + self.num_steady_state_steps:
                cmds.append(self._get_forward_step_task(step_id, graph_break=not self.fused_fwd_bwd))
                cmds.append(self._get_backward_step_task(step_id))
                # If this is the last step for steady state, do not recv fwd
                # since there is no further forward steps
                recv_fwd = step_id != (self.num_warmup_steps + self.num_steady_state_steps - 1)
                send_fwd = self._should_recv_bwd_send_fwd(step_id, is_forward=True)
                recv_bwd = self._should_recv_bwd_send_fwd(step_id, is_forward=False)
                send_bwd = self._should_recv_fwd_send_bwd(step_id, is_forward=False)
                self._add_pre_post_processing_tasks(
                    step_id, cmds, fwd_pre=recv_fwd, fwd_post=send_fwd, bwd_pre=recv_bwd, bwd_post=send_bwd
                )
            # Cool down
            else:
                cmds.append(self._get_backward_step_task(step_id))
                recv_fwd = False
                send_fwd = False
                recv_bwd = step_id != total_steps - 2 and self._should_recv_bwd_send_fwd(step_id, is_forward=False)
                send_bwd = self._should_recv_fwd_send_bwd(step_id, is_forward=False)
                self._add_pre_post_processing_tasks(
                    step_id, cmds, fwd_pre=recv_fwd, fwd_post=send_fwd, bwd_pre=recv_bwd, bwd_post=send_bwd
                )

            yield cmds


# Deprecated, kept for reference
class TrainSchedule(PipeSchedule):
    """
    1F1B schedule for training a batch using pipeline parallelism.
    Schedule is created by first assuming that even stage is doing 1F1B and odd stage is doing 1B1F
    without any idle time. Each step will be assigned with either fwd or bwd task. For the steps where
    a certain stage is supposed to be idle, we feed a invalid microbatch id to it. An example of
    pipeline exection with PP4 MB6 will be like
    Steps 0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17
    PP0   F0  IB  F1  IB  F2  IB  F3  B0  F4  B1  F5   B2   IF   B3   IF   B4   IF   B5
    PP1   IB  F0  IB  F1  IB  F2  B0  F3  B1  F4  B2   F5   B3   IF   B4   IF   B5   IF
    PP2   IF  IB  F0  IB  F1  B0  F2  B1  F3  B2  F4   B3   F5   B4   IF   B5   IF   IB
    PP3   IB  IF  IB  F0  B0  F1  B1  F2  B2  F3  B3   F4   B4   F5   B5   IF   IB   IF
    Where IB/IF means fake bwd/fwd with invalid microbatch ids, i.e. bubbles
    Referred from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/schedule.py#L189
    """

    def steps(self):
        prev_micro_batch_id = -1
        # 2 * self.num_microbatches for fwd+bwd steps
        # 2 * (self.stages - 1) for bubbles
        total_steps = 2 * (self.num_microbatches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)
            cmds: List[Union[PipelineTask, ReduceGradsTask]] = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(BackwardPostprocessTask(prev_micro_batch_id))
                if self._valid_micro_batch(micro_batch_id):
                    cmds.append(ForwardPreprocessTask(micro_batch_id))
            else:
                # Once enter the 1F1B steady states
                # Need to recv bwd before sending fwd to avoid hanging
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append(BackwardPreprocessTask(micro_batch_id))
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append(ForwardPostprocessTask(prev_micro_batch_id))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardStepTask(micro_batch_id))
                else:
                    cmds.append(BackwardStepTask(micro_batch_id))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceGradsTask())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    def _step_to_micro_batch(self, step_id):
        """
        This function calculate the microbatch id for each step based on the 1F1B schedule.
        The microbatch id is calculated from step_id, stage_id and total # stages
        For even stages, scheduler will assume it is doing 1F1B, so even step will be fwd otherwise bwd
        For odd stages, scheduler assumes 1B1F so even step will be bwd and odd step will be fwd
        The equation to calculate microbatch id:
            microbatch_id = base_microbatch_id - offset
            base_microbatch_id: microbatch id with the assumption that there is no invalid microbatches
            offset: the # invalid microbatches until the first valid microbatch
        Now we discuss how to calculate each for both forward and backward steps
        - forward steps
          - base = step_id // 2 as the nature of pipeline execution, base = (step_id-1) // 2 for
            odd stages to make it divisible
          - offset = stage_id // 2 which equals to the # extra mbs first stage need to run
            until current stage can start first valid mb
        - backward steps
          - base = step_id // 2 as the nature of pipeline execution, base = (step_id-1) // 2 for
            odd stages to make it divisible
          - offset = #invalid_forward_mbs + #warmup_mbs(i.e. # mbs need to run before 1F1B steady state)
            #warmup_mbs = #stage - stage_id - 1 (from #stage-1 to 0 starting from 0th stage)
            #invalid_forward_mbs = stage_id // 2 from fwd steps calculation
            But be careful, odd stages has an extra offset since it is starting with an invalid bwd step
        As a result
        - even step, even stage_id, fwd
            micro_batch_id = step_id // 2 - stage_id // 2
        - even step, odd stage_id, bwd
            micro_batch_id = step_id // 2 - (#stage - stage_id - 1) - stage_id // 2 - 1
              = step_id // 2 - #stage + stage_id + 1 - stage_id // 2 - 1
              = step_id // 2 - #stage + (stage_id + 1) // 2
        - odd step, even stage_id, bwd
            micro_batch_id = (step_id-1) // 2 - (#stage - stage_id - 1) - stage_id // 2
              = (step_id-1) // 2 - #stage + stage_id + 1 - stage_id // 2
              = (step_id-1) // 2 - #stage + stage_id // 2 + 1
        - odd step, odd stage_id, fwd
            micro_batch_id = (step_id-1) // 2 - stage_id // 2
        """
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stages + 1 + self.stage_id // 2)
        return micro_batch_id


def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0
