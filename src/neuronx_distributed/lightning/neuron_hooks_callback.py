from pytorch_lightning.callbacks import Callback
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers import parallel_state
import os
import numpy as np
import torch

class NeuronHooksCallback(Callback):
    def __init__(self, cfg):
        super().__init__()
        self.hooks = cfg.hooks
        if self.hooks:
            xm.master_print("Hooks are Active")
            self.activations_map = {}
            self.gradients_map = {}
            self.hooks_dump_base_directory = f"./hooks_outputs/{os.environ.get('SLURM_JOB_ID')}/hooks_dumps"
            self.master_print_model_layers = cfg.master_print_model_layers
            self.target_layers = [layer.strip() for layer in cfg.target_layers.split(",")]
            self.dump_only_master_rank = cfg.dump_only_master_rank
            self.dump_only_norms = cfg.dump_only_norms
            self.hooks_interval= cfg.hooks_interval
            self.enable_activation_dumps = cfg.enable_activation_dumps
            self.enable_grad_dumps = cfg.enable_grad_dumps
            self.enable_tb_logging_master_rank_activation_norms = cfg.enable_tb_logging_master_rank_activation_norms
            self.enable_tb_logging_master_rank_grad_norms = cfg.enable_tb_logging_master_rank_grad_norms
        else:
            self.activations_map = None
            self.gradients_map = None
            self.hooks_dump_base_directory = None
            self.master_print_model_layers = False
            self.target_layers = []
            self.dump_only_master_rank = None
            self.dump_only_norms = None
            self.hooks_interval = None
            self.enable_activation_dumps = False
            self.enable_grad_dumps = False
            self.enable_tb_logging_master_rank_activation_norms = False
            self.enable_tb_logging_master_rank_grad_norms = False


    def process_input_output(self, input, output):

        if isinstance(input, torch.Tensor):
            _input = input.detach().clone()
        elif isinstance(input, tuple):
            _input = input[0].detach().clone()

        if isinstance(output, torch.Tensor):
            _output = output.detach().clone()
        elif isinstance(output, tuple):
            _output = output[0].detach().clone()

        # move the tensors to CPU before doing anything downstream to have no impact on the graphs
        _input = _input.cpu()
        _output = _output.cpu()

        if self.dump_only_norms:
            # take L2 norms of torch.Tensors in i/o tensors
            _input = _input.norm().item()
            _output = _output.norm().item()

        return _input, _output


    def save_activations_map(self, pl_module):
        """
            Saves the recorded activations (layer input, layer output) for each target layer(s) in the model to host.
            Each activation is saved to a unique directory based on the layer name
            and global step. The activations are stored in .npy format.
            Function clears the activations map after saving to prevent re-saving the same tensors 
            on successive calls.
        """
        for layer_name, activation_list in self.activations_map.items():
            for activation_data in activation_list:
                global_step, layer_input, layer_output = activation_data[0], activation_data[1], activation_data[2]
                output_directory = f"{self.hooks_dump_base_directory}/{layer_name}/global_step_{global_step}"
                os.makedirs(output_directory, exist_ok=True)
                dp_pp_tp_config = parallel_state.get_rank_info_str()
                if self.dump_only_norms:
                    postfix = "norm"
                else:
                    postfix = "raw"
                torch.save(layer_input, f"{output_directory}/{dp_pp_tp_config}_layer_input_{postfix}.pt")
                torch.save(layer_output, f"{output_directory}/{dp_pp_tp_config}_layer_output_{postfix}.pt")       

                if self.dump_only_master_rank and self.dump_only_norms and self.enable_tb_logging_master_rank_activation_norms:
                    pl_module.log("activation_input_norm", layer_input, on_step=True, on_epoch=False, logger=True)
                    pl_module.log("activation_output_norm", layer_output, on_step=True, on_epoch=False, logger=True)

        self.activations_map.clear()

    def create_forward_hook(self, layer_name, pl_module):
        """
        Creates a forward hook function to record the input & output activations of target layer(s) during the forward pass.
        The activations are saved in a map using a unique identifier based on the layer name and module details.        
        """
        def hook(module, input, output):
            if pl_module.global_step % self.hooks_interval == 0:
                
                # Closure to process and save activations
                def process_and_save_activations():

                    if layer_name not in self.activations_map:
                        self.activations_map[layer_name] = []

                    processed_input, processed_output = self.process_input_output(input, output)
                    
                    # Conditional saving based on master rank
                    if (not self.dump_only_master_rank) or (self.dump_only_master_rank and xm.get_ordinal() == 0):
                        self.activations_map[layer_name].append((
                            pl_module.global_step,
                            processed_input,
                            processed_output
                        ))
                    
                    # Invoke materialization & saving of tensors after barrier
                    self.save_activations_map(pl_module)
                
                # Schedule the closure to run after the XLA step is completed
                xm.add_step_closure(process_and_save_activations)
        
        return hook

    def register_forward_hook_wrapper(self, layer_name, layer, pl_module):
        """
            Registers the forward hook function to the target layer.
            The hook will capture activations for the given layer during the forward pass.
        """
        layer.register_forward_hook(self.create_forward_hook(layer_name, pl_module))


    def save_gradients_map(self, pl_module):
        """
        Saves the recorded gradients for each target layer(s) in the model to host.
        Each gradient is saved to a unique directory based on the layer name and global step.
        The gradients are stored in .npy format.
        Function clears the gradients map after saving to prevent re-saving the same tensors
        on successive calls.
        """
        for layer_name, gradient_list in self.gradients_map.items():
            for gradient_data in gradient_list:
                global_step, grad_input, grad_output = (
                    gradient_data[0],
                    gradient_data[1],
                    gradient_data[2]
                )
                output_directory = f"{self.hooks_dump_base_directory}/{layer_name}/global_step_{global_step}"
                os.makedirs(output_directory, exist_ok=True)
                dp_pp_tp_config = parallel_state.get_rank_info_str()
                if self.dump_only_norms:
                    postfix = "norm"
                else:
                    postfix = "raw"
                torch.save(grad_input, f"{output_directory}/{dp_pp_tp_config}_grad_input_{postfix}.pt")
                torch.save(grad_output, f"{output_directory}/{dp_pp_tp_config}_grad_output_{postfix}.pt")
                if self.dump_only_master_rank and self.dump_only_norms and self.enable_tb_logging_master_rank_grad_norms:
                    pl_module.log("grad_input_norm", grad_input, on_step=True, on_epoch=False, logger=True)
                    pl_module.log("grad_output_norm", grad_output, on_step=True, on_epoch=False, logger=True)

        self.gradients_map.clear()

    def create_backward_hook(self, layer_name, pl_module):
        """
        Creates a backward hook function to record the gradients for target layer(s) during the backward pass.
        The gradients are saved in a map using a unique identifier based on the layer name and module details.
        """
        def hook(module, grad_input, grad_output):
            if pl_module.global_step % self.hooks_interval == 0:
                
                def process_and_save_gradients():

                    if layer_name not in self.gradients_map:
                        self.gradients_map[layer_name] = []
                        
                    processed_input, processed_output = self.process_input_output(grad_input, grad_output)
                    
                    if (not self.dump_only_master_rank) or (self.dump_only_master_rank and xm.get_ordinal() == 0):
                        self.gradients_map[layer_name].append((
                            pl_module.global_step,
                            processed_input,
                            processed_output
                        ))
                    
                    self.save_gradients_map(pl_module)
                
                xm.add_step_closure(process_and_save_gradients)
        
        return hook


    def register_backward_hook_wrapper(self, layer_name, layer, pl_module):
        """
        Registers the backward hook function to the target layer.
        The hook will capture gradients for the given layer during the backward pass.
        """
        layer.register_full_backward_hook(self.create_backward_hook(layer_name, pl_module))


    def on_train_start(self, trainer, pl_module):
        # Logic to print the model definition
        if self.master_print_model_layers:
            xm.master_print("Printing Model Layers:")
            for layer_name, layer in pl_module.model.named_modules():
                xm.master_print(f"{layer_name}")

        # Logic to hook the Activation/Grad Hooks to target layer(s)
        if len(self.target_layers)==0:
            pass
        else:
            self.target_layers = [s.strip() for s in self.target_layers]
            for layer_name, layer in pl_module.model.named_modules():
                layer_name = layer_name.strip()
                if layer_name in self.target_layers:
                    if self.enable_activation_dumps:
                        self.register_forward_hook_wrapper(layer_name, layer, pl_module)
                    if self.enable_grad_dumps:
                        self.register_backward_hook_wrapper(layer_name, layer, pl_module)
