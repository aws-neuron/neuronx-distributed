from abc import ABC, abstractmethod


class PipelineTask:
    def __init__(self, mb):
        self.mb = mb


class ForwardStepTask(PipelineTask):
    def __repr__(self):
        return f"ForwardStepTask_microbatch_{self.mb}"


class ForwardPreprocessTask(PipelineTask):
    def __repr__(self):
        return f"ForwardPreprocessTask_microbatch_{self.mb}"


class ForwardPostprocessTask(PipelineTask):
    def __repr__(self):
        return f"ForwardPostprocessTask_microbatch_{self.mb}"


class BackwardStepTask(PipelineTask):
    def __repr__(self):
        return f"BackwardStepTask_microbatch_{self.mb}"


class BackwardPreprocessTask(PipelineTask):
    def __repr__(self):
        return f"BackwardPreprocessTask_microbatch_{self.mb}"


class BackwardPostprocessTask(PipelineTask):
    def __repr__(self):
        return f"BackwardPostprocessTask_microbatch_{self.mb}"


class PostProcessTask:
    def __init__(self):
        """
        PostProcessTask happens after pipeline execution
        """
        self.mb = -1


class ReduceGradsTask(PostProcessTask):
    def __repr__(self):
        return f"ReduceGradsTask"


class PipeSchedule(ABC):
    """Directs the execution of a pipeline engine by generating sequences of
    :class:`PipelineTask`.

    Schedules are generators that yield sequences of
    :class:`PipelineTask` to process the micro-batches in one batch.
    Each yielded step is atomic in the sense that a barrier
    synchronization can be placed between successive steps without
    deadlock.

    Args:
        micro_batches (int): The number of micro-batches that comprise a batch.
        stages (int): The number of pipeline stages.
        stage_id (int): The pipe stage that will execute the generated schedule.
    """

    def __init__(self, micro_batches, stages, stage_id):
        super().__init__()
        self.micro_batches = micro_batches
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
        return 0 <= micro_batch_id < self.micro_batches

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
    def num_micro_batches(self):
        """The number of total micro_batches used to configure this schedule."""
        return self.micro_batches

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
        total_steps = self.micro_batches
        for micro_batch_id in range(total_steps):
            cmds = []
            cmds.append(ForwardPreprocessTask(micro_batch_id))
            cmds.append(ForwardStepTask(micro_batch_id))
            cmds.append(ForwardPostprocessTask(micro_batch_id))
            yield cmds


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
        # 2 * self.micro_batches for fwd+bwd steps
        # 2 * (self.stages - 1) for bubbles
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)
            cmds = []

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
