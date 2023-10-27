# Standard Library
import unittest
from unittest.mock import patch

# Third Party
from neuronx_distributed.pipeline.scheduler import TrainSchedule, InferenceSchedule, ForwardPreprocessTask \
        , ForwardStepTask, ForwardPostprocessTask, BackwardPreprocessTask, BackwardStepTask, BackwardPostprocessTask, ReduceGradsTask
from .. import update_result


class TestScheduler(unittest.TestCase):
    
    @patch('torch.distributed.get_rank') 
    def test_train_schedule_mb2_stage0(self, rank_mock):
        try:
            train_scheduler = TrainSchedule(micro_batches=2, stages=4, stage_id=0)
            tasks = [task for task in train_scheduler.steps()]
            expected_tasks = [[ForwardPreprocessTask(mb=0), ForwardStepTask(mb=0)], [ForwardPostprocessTask(mb=0)], 
                            [ForwardPreprocessTask(mb=1), ForwardStepTask(mb=1)], [ForwardPostprocessTask(mb=1)], 
                            [], [], [], [BackwardPreprocessTask(mb=0), BackwardStepTask(mb=0)], [], 
                            [BackwardPreprocessTask(mb=1), BackwardStepTask(mb=1), ReduceGradsTask()]]
            assert str(tasks) == str(expected_tasks)
        except:
            update_result({"inference_success": 0})
            raise

    @patch('torch.distributed.get_rank') 
    def test_train_schedule_mb2_stage1(self, rank_mock):
        try:
            train_scheduler = TrainSchedule(micro_batches=2, stages=4, stage_id=1)
            tasks = [task for task in train_scheduler.steps()]
            expected_tasks = [[], [ForwardPreprocessTask(mb=0), ForwardStepTask(mb=0)], [ForwardPostprocessTask(mb=0)], 
                            [ForwardPreprocessTask(mb=1), ForwardStepTask(mb=1)], [ForwardPostprocessTask(mb=1)], [], 
                            [BackwardPreprocessTask(mb=0), BackwardStepTask(mb=0)], [BackwardPostprocessTask(mb=0)], 
                            [BackwardPreprocessTask(mb=1), BackwardStepTask(mb=1)], [BackwardPostprocessTask(mb=1), ReduceGradsTask()]]
            assert str(tasks) == str(expected_tasks)
        except:
            update_result({"inference_success": 0})
            raise

    @patch('torch.distributed.get_rank') 
    def test_train_schedule_mb2_stage2(self, rank_mock):
        try:
            train_scheduler = TrainSchedule(micro_batches=2, stages=4, stage_id=2)
            tasks = [task for task in train_scheduler.steps()]
            expected_tasks = [[], [], [ForwardPreprocessTask(mb=0), ForwardStepTask(mb=0)], 
                            [ForwardPostprocessTask(mb=0)], [ForwardPreprocessTask(mb=1), ForwardStepTask(mb=1)], 
                            [BackwardPreprocessTask(mb=0), ForwardPostprocessTask(mb=1), BackwardStepTask(mb=0)], 
                            [BackwardPostprocessTask(mb=0)], [BackwardPreprocessTask(mb=1), BackwardStepTask(mb=1)], 
                            [BackwardPostprocessTask(mb=1)], [ReduceGradsTask()]]
            assert str(tasks) == str(expected_tasks)
        except:
            update_result({"inference_success": 0})
            raise

    @patch('torch.distributed.get_rank') 
    def test_train_schedule_mb2_stage3(self, rank_mock):
        try:
            train_scheduler = TrainSchedule(micro_batches=2, stages=4, stage_id=3)
            tasks = [task for task in train_scheduler.steps()]
            expected_tasks = [[], [], [], [ForwardPreprocessTask(mb=0), ForwardStepTask(mb=0)], 
                            [BackwardStepTask(mb=0)], [BackwardPostprocessTask(mb=0), ForwardPreprocessTask(mb=1), ForwardStepTask(mb=1)], 
                            [BackwardStepTask(mb=1)], [BackwardPostprocessTask(mb=1)], [], [ReduceGradsTask()]]
            assert str(tasks) == str(expected_tasks)
        except:
            update_result({"inference_success": 0})
            raise

    @patch('torch.distributed.get_rank') 
    def test_train_schedule_mb8_stage0(self, rank_mock):
        try:
            train_scheduler = TrainSchedule(micro_batches=8, stages=4, stage_id=0)
            tasks = [task for task in train_scheduler.steps()]
            expected_tasks = [[ForwardPreprocessTask(mb=0), ForwardStepTask(mb=0)], [ForwardPostprocessTask(mb=0)], 
                            [ForwardPreprocessTask(mb=1), ForwardStepTask(mb=1)], [ForwardPostprocessTask(mb=1)], 
                            [ForwardPreprocessTask(mb=2), ForwardStepTask(mb=2)], [ForwardPostprocessTask(mb=2)], 
                            [ForwardPreprocessTask(mb=3), ForwardStepTask(mb=3)], 
                            [BackwardPreprocessTask(mb=0), ForwardPostprocessTask(mb=3), BackwardStepTask(mb=0)], 
                            [ForwardPreprocessTask(mb=4), ForwardStepTask(mb=4)], 
                            [BackwardPreprocessTask(mb=1), ForwardPostprocessTask(mb=4), BackwardStepTask(mb=1)], 
                            [ForwardPreprocessTask(mb=5), ForwardStepTask(mb=5)], 
                            [BackwardPreprocessTask(mb=2), ForwardPostprocessTask(mb=5), BackwardStepTask(mb=2)], 
                            [ForwardPreprocessTask(mb=6), ForwardStepTask(mb=6)], 
                            [BackwardPreprocessTask(mb=3), ForwardPostprocessTask(mb=6), BackwardStepTask(mb=3)], 
                            [ForwardPreprocessTask(mb=7), ForwardStepTask(mb=7)], 
                            [BackwardPreprocessTask(mb=4), ForwardPostprocessTask(mb=7), BackwardStepTask(mb=4)], [], 
                            [BackwardPreprocessTask(mb=5), BackwardStepTask(mb=5)], [], 
                            [BackwardPreprocessTask(mb=6), BackwardStepTask(mb=6)], [], 
                            [BackwardPreprocessTask(mb=7), BackwardStepTask(mb=7), ReduceGradsTask()]]
        
            assert str(tasks) == str(expected_tasks)
        except:
            update_result({"inference_success": 0})
            raise

    @patch('torch.distributed.get_rank') 
    def test_train_schedule_mb8_stage1(self, rank_mock):
        try:
            train_scheduler = TrainSchedule(micro_batches=8, stages=4, stage_id=1)
            tasks = [task for task in train_scheduler.steps()]
            expected_tasks = [[], [ForwardPreprocessTask(mb=0), ForwardStepTask(mb=0)], [ForwardPostprocessTask(mb=0)], 
                            [ForwardPreprocessTask(mb=1), ForwardStepTask(mb=1)], [ForwardPostprocessTask(mb=1)], 
                            [ForwardPreprocessTask(mb=2), ForwardStepTask(mb=2)], 
                            [BackwardPreprocessTask(mb=0), ForwardPostprocessTask(mb=2), BackwardStepTask(mb=0)], 
                            [BackwardPostprocessTask(mb=0), ForwardPreprocessTask(mb=3), ForwardStepTask(mb=3)], 
                            [BackwardPreprocessTask(mb=1), ForwardPostprocessTask(mb=3), BackwardStepTask(mb=1)], 
                            [BackwardPostprocessTask(mb=1), ForwardPreprocessTask(mb=4), ForwardStepTask(mb=4)], 
                            [BackwardPreprocessTask(mb=2), ForwardPostprocessTask(mb=4), BackwardStepTask(mb=2)], 
                            [BackwardPostprocessTask(mb=2), ForwardPreprocessTask(mb=5), ForwardStepTask(mb=5)], 
                            [BackwardPreprocessTask(mb=3), ForwardPostprocessTask(mb=5), BackwardStepTask(mb=3)], 
                            [BackwardPostprocessTask(mb=3), ForwardPreprocessTask(mb=6), ForwardStepTask(mb=6)], 
                            [BackwardPreprocessTask(mb=4), ForwardPostprocessTask(mb=6), BackwardStepTask(mb=4)], 
                            [BackwardPostprocessTask(mb=4), ForwardPreprocessTask(mb=7), ForwardStepTask(mb=7)], 
                            [BackwardPreprocessTask(mb=5), ForwardPostprocessTask(mb=7), BackwardStepTask(mb=5)], 
                            [BackwardPostprocessTask(mb=5)], [BackwardPreprocessTask(mb=6), BackwardStepTask(mb=6)], 
                            [BackwardPostprocessTask(mb=6)], [BackwardPreprocessTask(mb=7), BackwardStepTask(mb=7)], 
                            [BackwardPostprocessTask(mb=7), ReduceGradsTask()]]
            assert str(tasks) == str(expected_tasks)
        except:
            update_result({"inference_success": 0})
            raise

    @patch('torch.distributed.get_rank') 
    def test_train_schedule_mb8_stage2(self, rank_mock):
        try:
            train_scheduler = TrainSchedule(micro_batches=8, stages=4, stage_id=2)
            tasks = [task for task in train_scheduler.steps()]
            expected_tasks = [[], [], [ForwardPreprocessTask(mb=0), ForwardStepTask(mb=0)], [ForwardPostprocessTask(mb=0)], 
                            [ForwardPreprocessTask(mb=1), ForwardStepTask(mb=1)], 
                            [BackwardPreprocessTask(mb=0), ForwardPostprocessTask(mb=1), BackwardStepTask(mb=0)], 
                            [BackwardPostprocessTask(mb=0), ForwardPreprocessTask(mb=2), ForwardStepTask(mb=2)], 
                            [BackwardPreprocessTask(mb=1), ForwardPostprocessTask(mb=2), BackwardStepTask(mb=1)], 
                            [BackwardPostprocessTask(mb=1), ForwardPreprocessTask(mb=3), ForwardStepTask(mb=3)], 
                            [BackwardPreprocessTask(mb=2), ForwardPostprocessTask(mb=3), BackwardStepTask(mb=2)], 
                            [BackwardPostprocessTask(mb=2), ForwardPreprocessTask(mb=4), ForwardStepTask(mb=4)], 
                            [BackwardPreprocessTask(mb=3), ForwardPostprocessTask(mb=4), BackwardStepTask(mb=3)], 
                            [BackwardPostprocessTask(mb=3), ForwardPreprocessTask(mb=5), ForwardStepTask(mb=5)], 
                            [BackwardPreprocessTask(mb=4), ForwardPostprocessTask(mb=5), BackwardStepTask(mb=4)], 
                            [BackwardPostprocessTask(mb=4), ForwardPreprocessTask(mb=6), ForwardStepTask(mb=6)], 
                            [BackwardPreprocessTask(mb=5), ForwardPostprocessTask(mb=6), BackwardStepTask(mb=5)], 
                            [BackwardPostprocessTask(mb=5), ForwardPreprocessTask(mb=7), ForwardStepTask(mb=7)], 
                            [BackwardPreprocessTask(mb=6), ForwardPostprocessTask(mb=7), BackwardStepTask(mb=6)], 
                            [BackwardPostprocessTask(mb=6)], [BackwardPreprocessTask(mb=7), BackwardStepTask(mb=7)], 
                            [BackwardPostprocessTask(mb=7)], [ReduceGradsTask()]]
            assert str(tasks) == str(expected_tasks)
        except:
            update_result({"inference_success": 0})
            raise

    @patch('torch.distributed.get_rank') 
    def test_train_schedule_mb8_stage3(self, rank_mock):
        try:
            train_scheduler = TrainSchedule(micro_batches=8, stages=4, stage_id=3)
            tasks = [task for task in train_scheduler.steps()]
            expected_tasks = [[], [], [], [ForwardPreprocessTask(mb=0), ForwardStepTask(mb=0)], [BackwardStepTask(mb=0)], 
                            [BackwardPostprocessTask(mb=0), ForwardPreprocessTask(mb=1), ForwardStepTask(mb=1)], 
                            [BackwardStepTask(mb=1)], [BackwardPostprocessTask(mb=1), ForwardPreprocessTask(mb=2), ForwardStepTask(mb=2)], 
                            [BackwardStepTask(mb=2)], [BackwardPostprocessTask(mb=2), ForwardPreprocessTask(mb=3), ForwardStepTask(mb=3)], 
                            [BackwardStepTask(mb=3)], [BackwardPostprocessTask(mb=3), ForwardPreprocessTask(mb=4), ForwardStepTask(mb=4)], 
                            [BackwardStepTask(mb=4)], [BackwardPostprocessTask(mb=4), ForwardPreprocessTask(mb=5), ForwardStepTask(mb=5)], 
                            [BackwardStepTask(mb=5)], [BackwardPostprocessTask(mb=5), ForwardPreprocessTask(mb=6), ForwardStepTask(mb=6)], 
                            [BackwardStepTask(mb=6)], [BackwardPostprocessTask(mb=6), ForwardPreprocessTask(mb=7), ForwardStepTask(mb=7)], 
                            [BackwardStepTask(mb=7)], [BackwardPostprocessTask(mb=7)], [], [ReduceGradsTask()]]
            assert str(tasks) == str(expected_tasks)
        except:
            update_result({"inference_success": 0})
            raise

    @patch('torch.distributed.get_rank') 
    def test_inference_schedule_mb2_stage4(self, rank_mock):
        try:
            inference_scheduler = InferenceSchedule(micro_batches=2, stages=4, stage_id=2)
            tasks = [task for task in inference_scheduler.steps()]
            expected_tasks = [[ForwardPreprocessTask(mb=0), ForwardStepTask(mb=0), ForwardPostprocessTask(mb=0)], 
                            [ForwardPreprocessTask(mb=1), ForwardStepTask(mb=1), ForwardPostprocessTask(mb=1)]]
            assert str(tasks) == str(expected_tasks)
        except:
            update_result({"inference_success": 0})
            raise


if __name__ == "__main__":
    unittest.main()
    