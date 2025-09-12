import os
import unittest
import shutil
import torch
import torch.nn as nn

from neuronx_distributed.utils.model_utils import init_on_device
from neuronx_distributed import NxDParallelState, shard_checkpoint, ModelBuilder
from neuronx_distributed.trace.model_builder_utils import ModelBuilderConstants
from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear

torch.manual_seed(0)

class TestE2EModelBuilderV2Flow(unittest.TestCase):
    def test_e2e_with_priority_model(self):

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(1024, 1024)
                self.layer2 = nn.Linear(1024, 1024)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)

        cpu_model = Model()
        model_checkpoint = cpu_model.state_dict()

        model = Model()

        example_inputs1 = torch.rand(32, 1024)
        example_inputs2 = torch.rand(16, 1024)
        
        nxd_model = ModelBuilder(model) \
            .trace(args=example_inputs1, tag="priority") \
            .trace(args=example_inputs2, tag="secondary") \
            .compile(priority_model_key="priority")

        nxd_model.set_weights([model_checkpoint])
        nxd_model.to_neuron()

        input1 = torch.rand(32, 1024)
        input2 = torch.rand(16, 1024)

        for input in [input1, input2]:
            cpu_out = cpu_model(input)
            neuron_out = nxd_model(input)
            torch.testing.assert_close(cpu_out, neuron_out)

        for key in ["priority", "secondary"]:
            shutil.rmtree(os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, key))

        torch.classes.neuron.Runtime().unsafe_close()

    def test_e2e_without_priority_model(self):

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(1024, 1024)
                self.layer2 = nn.Linear(1024, 1024)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)

        cpu_model = Model()
        model_checkpoint = cpu_model.state_dict()

        model = Model()

        example_inputs1 = torch.rand(32, 1024)
        example_inputs2 = torch.rand(16, 1024)
        
        nxd_model = ModelBuilder(model) \
            .trace(args=example_inputs1, tag="priority") \
            .trace(args=example_inputs2, tag="secondary") \
            .compile()

        nxd_model.set_weights([model_checkpoint])
        nxd_model.to_neuron()

        input1 = torch.rand(32, 1024)
        input2 = torch.rand(16, 1024)

        for input in [input1, input2]:
            cpu_out = cpu_model(input)
            neuron_out = nxd_model(input)
            torch.testing.assert_close(cpu_out, neuron_out)

        for key in ["priority", "secondary"]:
            shutil.rmtree(os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, key))
        
        torch.classes.neuron.Runtime().unsafe_close()


    def test_e2e_with_kwargs(self):

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(5, 10)
                self.linear2 = nn.Linear(20, 10)

            def forward(self, x, y):
                return self.linear1(x) + self.linear2(y)

        cpu_model = Model()
        model_checkpoint = cpu_model.state_dict()

        model = Model()

        example_inputs1 = {'x': torch.rand(10, 5), 'y': torch.rand(10, 20)}
        example_inputs2 = {'x': torch.rand(50, 5), 'y': torch.rand(50, 20)}
        
        nxd_model = ModelBuilder(model) \
            .trace(kwargs=example_inputs1, tag="priority") \
            .trace(kwargs=example_inputs2, tag="secondary") \
            .compile(priority_model_key="priority")

        nxd_model.set_weights([model_checkpoint])
        nxd_model.to_neuron()

        input1 = (torch.rand(10, 5), torch.rand(10, 20))
        input2 =  (torch.rand(50, 5), torch.rand(50, 20))

        for input in [input1, input2]:
            cpu_out = cpu_model(input[0], input[1])
            neuron_out = nxd_model(x=input[0], y=input[1])
            torch.testing.assert_close(cpu_out, neuron_out)

        for key in ["priority", "secondary"]:
            shutil.rmtree(os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, key))

        torch.classes.neuron.Runtime().unsafe_close()


class TestE2EModelBuilderV2FlowDistributed(unittest.TestCase):
    def test_e2e_with_priority_model(self):

        class Model(nn.Module):
            def __init__(self, is_distributed=True):
                super().__init__()
                if is_distributed:
                    self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
                    self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)
                else:
                    self.layer1 = nn.Linear(1024, 1024)
                    self.layer2 = nn.Linear(1024, 1024)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)

        cpu_model = Model(is_distributed=False)
        model_checkpoint = cpu_model.state_dict()

        with NxDParallelState(world_size=2, tensor_model_parallel_size=2):
            model = Model()

            example_inputs1 = torch.rand(32, 1024)
            example_inputs2 = torch.rand(16, 1024)
            
            nxd_model = ModelBuilder(model) \
                .trace(args=example_inputs1, tag="priority") \
                .trace(args=example_inputs2, tag="secondary") \
                .compile(priority_model_key="priority")


        with NxDParallelState(world_size=2, tensor_model_parallel_size=2), init_on_device(torch.device("meta")):
            sharded_checkpoint = shard_checkpoint(
                checkpoint=model_checkpoint,
                model=Model()
            )

        nxd_model.set_weights(sharded_checkpoint)
        nxd_model.to_neuron()

        input1 = torch.rand(32, 1024)
        input2 = torch.rand(16, 1024)

        for input in [input1, input2]:
            cpu_out = cpu_model(input)
            neuron_out = nxd_model(input)
            torch.testing.assert_close(cpu_out, neuron_out)

        for key in ["priority", "secondary"]:
            shutil.rmtree(os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, key))

        torch.classes.neuron.Runtime().unsafe_close()

    def test_e2e_without_priority_model(self):

        class Model(nn.Module):
            def __init__(self, is_distributed=True):
                super().__init__()
                if is_distributed:
                    self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
                    self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)
                else:
                    self.layer1 = nn.Linear(1024, 1024)
                    self.layer2 = nn.Linear(1024, 1024)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)

        cpu_model = Model(is_distributed=False)
        model_checkpoint = cpu_model.state_dict()

        with NxDParallelState(world_size=2, tensor_model_parallel_size=2):
            model = Model()

            example_inputs1 = torch.rand(32, 1024)
            example_inputs2 = torch.rand(16, 1024)
            
            nxd_model = ModelBuilder(model) \
                .trace(args=example_inputs1, tag="priority") \
                .trace(args=example_inputs2, tag="secondary") \
                .compile()


        with NxDParallelState(world_size=2, tensor_model_parallel_size=2), init_on_device(torch.device("meta")):
            sharded_checkpoint = shard_checkpoint(
                checkpoint=model_checkpoint,
                model=Model()
            )

        nxd_model.set_weights(sharded_checkpoint)
        nxd_model.to_neuron()

        input1 = torch.rand(32, 1024)
        input2 = torch.rand(16, 1024)

        for input in [input1, input2]:
            cpu_out = cpu_model(input)
            neuron_out = nxd_model(input)
            torch.testing.assert_close(cpu_out, neuron_out)

        for key in ["priority", "secondary"]:
            shutil.rmtree(os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, key))
        
        torch.classes.neuron.Runtime().unsafe_close()


    def test_e2e_with_kwargs(self):

        class Model(nn.Module):
            def __init__(self, is_distributed=True):
                super().__init__()
                if is_distributed:
                    self.linear1 = ColumnParallelLinear(5, 10, gather_output=True)
                    self.linear2 = ColumnParallelLinear(20, 10, gather_output=True)
                else:
                    self.linear1 = nn.Linear(5, 10)
                    self.linear2 = nn.Linear(20, 10)

            def forward(self, x, y):
                return self.linear1(x) + self.linear2(y)

        cpu_model = Model(is_distributed=False)
        model_checkpoint = cpu_model.state_dict()

        with NxDParallelState(world_size=2, tensor_model_parallel_size=2):
            model = Model()

            example_inputs1 = {'x': torch.rand(10, 5), 'y': torch.rand(10, 20)}
            example_inputs2 = {'x': torch.rand(50, 5), 'y': torch.rand(50, 20)}
            
            nxd_model = ModelBuilder(model) \
                .trace(kwargs=example_inputs1, tag="priority") \
                .trace(kwargs=example_inputs2, tag="secondary") \
                .compile(priority_model_key="priority")


        with NxDParallelState(world_size=2, tensor_model_parallel_size=2), init_on_device(torch.device("meta")):
            sharded_checkpoint = shard_checkpoint(
                checkpoint=model_checkpoint,
                model=Model()
            )

        nxd_model.set_weights(sharded_checkpoint)
        nxd_model.to_neuron()

        input1 = (torch.rand(10, 5), torch.rand(10, 20))
        input2 = (torch.rand(50, 5), torch.rand(50, 20))

        for input in [input1, input2]:
            cpu_out = cpu_model(input[0], input[1])
            neuron_out = nxd_model(x=input[0], y=input[1])
            torch.testing.assert_close(cpu_out, neuron_out)

        for key in ["priority", "secondary"]:
            shutil.rmtree(os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, key))

        torch.classes.neuron.Runtime().unsafe_close()

if __name__ == '__main__':
    unittest.main()
