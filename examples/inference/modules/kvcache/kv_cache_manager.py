import torch
from torch import nn, Tensor

from modules.autobucketing import slice_lhs, slice_rhs
from modules.config import NeuronConfig
from neuronx_distributed.parallel_layers import parallel_state, utils
from modules.gqa import (  # noqa: E402
    determine_sharding_strategy,  # noqa: E402
    get_shardable_head_counts,  # noqa: E402
)  # noqa: E402

from typing import Dict, List


def _slice_kv_cacheline(padding_side, seq_len: int, cache: Tensor):
    if padding_side == "right":
        return slice_lhs(cache, seq_len, 2)
    else:
        max_idx = cache.shape[2]
        return slice_rhs(cache, seq_len, max_idx, 2)


def _gather_slice_into_kv_cacheline(cache, padding_side, seq_len: int, bucket_slice: Tensor):
    max_idx = cache.shape[2]
    if padding_side == "right":
        remaining = slice_rhs(cache, max_idx - seq_len, max_idx, dim=2)
        return torch.cat([bucket_slice, remaining], dim=2)
    else:
        remaining = slice_lhs(cache, max_idx - seq_len, dim=2)
        return torch.cat([remaining, bucket_slice], dim=2)


class KVCacheManager(nn.Module):
    """
    Key Value Cache Management.
    It stores KV cache as a parameter list of the shape (batch_sz, num_kv_head_per_partition, max_len, hidden_dim),
    and vends out read and write operations.
    """
    def __init__(self, neuron_config: NeuronConfig, **kwargs):

        super().__init__()
        self.is_medusa = neuron_config.is_medusa
        self.num_medusa_heads = neuron_config.num_medusa_heads
        self.padding_side = neuron_config.padding_side
        self.is_continuous_batching = neuron_config.is_continuous_batching
        self.flash_decoding_enabled = neuron_config.flash_decoding_enabled
        self.num_cores_per_group = neuron_config.num_cores_per_group if neuron_config.flash_decoding_enabled else None
        self.num_kv_head = kwargs['num_kv_head']
        self._init_kv_shape(neuron_config)

        num_layer = neuron_config.hf_config.num_hidden_layers
        dtype = neuron_config.hf_config.torch_dtype
        self.past_key_values = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.kv_shape, dtype=dtype), requires_grad=False)
                for _ in range(num_layer * 2)
            ]
        )

    def _init_kv_shape(self, neuron_config):
        max_batch_size = neuron_config.max_batch_size
        max_len = neuron_config.max_length
        tp_degree = neuron_config.tp_degree
        num_kv_head = self.num_kv_head
        num_atten_head = neuron_config.hf_config.num_attention_heads
        hidden_size = neuron_config.hf_config.hidden_size

        gqa_sharding_strategy = determine_sharding_strategy(tp_degree, num_kv_head)
        _, num_key_value_heads = get_shardable_head_counts(
            tp_degree, num_atten_head, num_kv_head, gqa_sharding_strategy
        )

        if parallel_state.model_parallel_is_initialized():
            num_kv_heads_per_partition = utils.divide(num_key_value_heads, tp_degree)
        else:
            num_kv_heads_per_partition = num_key_value_heads
        hidden_dim_per_head = hidden_size // num_atten_head

        if self.flash_decoding_enabled:
            max_len = utils.divide(max_len, self.num_cores_per_group)

        self.kv_shape = (
            max_batch_size,
            num_kv_heads_per_partition,
            max_len,
            hidden_dim_per_head,
        )

    def configure_medusa_gather_slice_idx(self, metadata):
        assert 'current_length' in metadata and 'accepted_indices' in metadata, \
            'current_length and accepted_indices should be specified for medusa decoding!'

        current_length = metadata['current_length']
        accepted_indices = metadata['accepted_indices']
        slice_index = current_length.view(-1, 1, current_length.shape[-1], 1).expand_as(
            self.past_key_values[0][:, :, 0: self.num_medusa_heads + 1, :]
        )
        gather_index = accepted_indices.view(-1, 1, accepted_indices.shape[-1], 1).expand_as(
            self.past_key_values[0][:, :, 0: self.num_medusa_heads + 1, :])
        return slice_index, gather_index

    def get_kv_by_layer_id(self, key_layer_idx, gather_index=None, slice_index=None):
        k_cache = self.past_key_values[key_layer_idx]
        v_cache = self.past_key_values[key_layer_idx + 1]
        if self.is_medusa:
            accepted_k_cache = torch.gather(input=k_cache, dim=2, index=gather_index)
            accepted_v_cache = torch.gather(input=v_cache, dim=2, index=gather_index)
            k_cache = torch.scatter(input=k_cache, dim=2, index=slice_index, src=accepted_k_cache)
            v_cache = torch.scatter(input=v_cache, dim=2, index=slice_index, src=accepted_v_cache)
        return k_cache, v_cache

    def get_cache(self, seq_len: int, **kwargs):
        """
        Return network (all layers)'s previously cached K and V, up to seq_len.

        :param seq_len: sequence length (or bucket size from auto-bucketing e.g. 128, 512, 1024 etc.)
        :return: list of tuple of (K, V)
        """
        slice_index, gather_index = None, None
        if self.is_medusa:
            assert 'medusa_metadata' in kwargs, 'medusa_metadata should be specified for medusa decoding!'
            medusa_metadata = kwargs['medusa_metadata']
            slice_index, gather_index = self.configure_medusa_gather_slice_idx(medusa_metadata)
        past_key_values = []
        for key_layer_idx in range(0, len(self.past_key_values), 2):
            # get kv per layer
            k_cache, v_cache = self.get_kv_by_layer_id(key_layer_idx, gather_index=gather_index, slice_index=slice_index)
            # slice for partial view
            key_state = _slice_kv_cacheline(self.padding_side, seq_len, k_cache)
            value_state = _slice_kv_cacheline(self.padding_side, seq_len, v_cache)
            past_key_values.append([key_state, value_state])
        return past_key_values

    def update_cache(self, is_for_context_encoding: bool, seq_ids: Tensor, position_ids: Tensor,
                     past_key_values: List[Tensor], seq_len: int, scatter_index=None, active_mask=None):
        """
        Given the passed-in past_key_values, update the cache

        :param scatter_index: tensor representing index to update
        :param is_for_context_encoding: bool
        :param seq_ids: tensor of size (batch_sz)
        :param position_ids: tensor of size (batch_sz, bucket_sz)
        :param past_key_values: list of tuple, the latest kv obtained at the end of the network from forward pass
        :param seq_len: sequence length (or bucket size from auto-bucketing e.g. 128, 512, 1024 etc.)
        :param scatter_index: tensor representing index to update
        :param active_mask: tensor representing index to update
        :return: list of tuple of (K, V)
        """
        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(past_key_values):

            latest_k, latest_v = kv_per_layer[0], kv_per_layer[1]

            # read differently based on padding side and bucket sz
            k_cache = _slice_kv_cacheline(self.padding_side, seq_len, self.past_key_values[idx * 2])
            v_cache = _slice_kv_cacheline(self.padding_side, seq_len, self.past_key_values[idx * 2 + 1])

            if is_for_context_encoding:
                if self.is_continuous_batching:
                    # scatter back to the desired seq_ids
                    seq_id_index_shape = seq_ids.shape[:1] + k_cache.shape[1:]
                    seq_id_index = seq_ids.view(-1, 1, 1, 1).expand(seq_id_index_shape)
                    k_cache = torch.scatter(input=k_cache, dim=0, index=seq_id_index, src=latest_k)
                    v_cache = torch.scatter(input=v_cache, dim=0, index=seq_id_index, src=latest_v)
                else:
                    # assign back to full kv_cacheline
                    k_cache = latest_k
                    v_cache = latest_v
            else:
                if self.padding_side == "left":
                    # TODO: fix it with scatter after right padding
                    k_cache = k_cache[:, :, 1:, :]
                    v_cache = v_cache[:, :, 1:, :]
                    k_cache = torch.cat([k_cache, latest_k], dim=2)
                    v_cache = torch.cat([v_cache, latest_v], dim=2)
                else:
                    # copy the tensor of the new position into kv cache
                    if self.flash_decoding_enabled:
                        assert active_mask is not None, 'active_mask should be specified for flash decoding!'
                        garbage_pos = seq_len-1  # treat last pos as garbage
                        updated_pos_ids = position_ids // self.num_cores_per_group
                        scatter_index = torch.where(active_mask == 1, updated_pos_ids, garbage_pos)
                        scatter_index = scatter_index.view(-1, 1, scatter_index.shape[-1], 1).expand_as(latest_k)
                    else:
                        scatter_index = self._get_index_to_update_new_position(scatter_index, position_ids, latest_k)

                    k_cache = torch.scatter(input=k_cache, dim=2, index=scatter_index, src=latest_k)
                    v_cache = torch.scatter(input=v_cache, dim=2, index=scatter_index, src=latest_v)

            # update
            k_cache = _gather_slice_into_kv_cacheline(self.past_key_values[idx * 2], self.padding_side, seq_len, k_cache)
            v_cache = _gather_slice_into_kv_cacheline(self.past_key_values[idx * 2 + 1], self.padding_side, seq_len,
                                                      v_cache)
            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        return updated_kv_cache

    def _get_index_to_update_new_position(self, scatter_index, position_ids, full_k):
        if self.is_medusa and scatter_index is not None:
            scatter_index = scatter_index.view(-1, 1, scatter_index.shape[-1], 1).expand_as(full_k)
        else:
            scatter_index = position_ids.view(-1, 1, position_ids.shape[-1], 1).expand_as(full_k)
        return scatter_index
