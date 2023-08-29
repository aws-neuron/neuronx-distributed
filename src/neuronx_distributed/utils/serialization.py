# Standard Library
import collections
import copy
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, NamedTuple

# Third Party
import torch


def is_instance_namedtuple(iterable):
    return isinstance(iterable, tuple) and iterable.__class__.__base__ == tuple and hasattr(iterable, "_fields")


def find_loss_from_output_and_spec(output_val, spec_val):
    """
    Fetch loss tensor from the output spec provided by user.
    Refered from https://github.com/pytorch/PiPPy/blob/main/pippy/IR.py#L42
    """
    if spec_val is False:
        return None
    if spec_val is True:
        if not isinstance(output_val, torch.Tensor):
            raise RuntimeError(f"Loss spec must specify a tensor value but got {output_val}")
        return output_val

    if isinstance(spec_val, (tuple, list)):
        if not isinstance(output_val, (tuple, list)):
            raise RuntimeError(f"Output value {output_val} must match type of loss specification " f"{spec_val}")
        if len(output_val) != len(spec_val):
            raise RuntimeError(f"Output value {output_val} must match length of loss specification " f"{spec_val}")
        for out, spec in zip(output_val, spec_val):
            loss_val = find_loss_from_output_and_spec(out, spec)
            if loss_val is not None:
                return loss_val
        raise RuntimeError(f"Did not find loss value in specification {spec_val}")

    if isinstance(spec_val, dict):
        if not isinstance(output_val, dict):
            raise RuntimeError(f"Output value {output_val} must match type of loss specification " f"{spec_val}")
        if set(output_val.keys()) != set(spec_val.keys()):
            raise RuntimeError(f"Output value {output_val} must match keys of loss specification " f"{spec_val}")
        for k in spec_val:
            loss_val = find_loss_from_output_and_spec(output_val[k], spec_val[k])
            if loss_val is not None:
                return loss_val
        raise RuntimeError(f"Did not find loss value in specification {spec_val}")

    raise RuntimeError(f"Unsupported type {type(spec_val)} in loss specification")


class TensorMeta(NamedTuple):
    """
    Represents a tensor in serialized messages. Can be used at the destination to reconstruct
    the original tensor.
    """

    tensor_index: int
    dtype: torch.dtype
    shape: torch.Size
    requires_grad: bool
    device: torch.device


class SerializationManager:
    """
    Extract the tensors and the python object that holds them. Currently only support tuple, list, set and dict
    Usage:
        # To serialize an object
        py_obj, tensor_list, tensor_meta_list = SerializationManager.serialize(obj)
        # To deserialize an object
        obj = deserialize(py_obj, tensor_list)
    """

    @contextmanager
    def catch_and_raise_for_large_object(self, obj):
        try:
            yield
        except RecursionError:
            raise RuntimeError(obj.__class__.__name__)

    def serialize(
        self,
        obj: Any,
        return_stub_list: bool = True,
    ):
        tx_list = []
        stub_list = []
        with self.catch_and_raise_for_large_object(obj):
            obj_stripped_of_tensors, seen_class_types = self._replace_tensors_with_stubs(obj, {}, tx_list, stub_list)

            # the above will mutate the class types in-place, so if we have encountered any non-Tensor
            # class types, we need to deepcopy the serialized object for transmission, and then reconstruct
            # the original object locally
            # [TODO] Do we really need to reconstruct the original object?
            if seen_class_types:
                serialized_cpy = copy.deepcopy(obj_stripped_of_tensors)
                self.deserialize(obj_stripped_of_tensors, tx_list)

                if return_stub_list:
                    return serialized_cpy, tx_list, stub_list
                else:
                    return serialized_cpy, tx_list
            else:
                if return_stub_list:
                    return obj_stripped_of_tensors, tx_list, stub_list
                else:
                    return obj_stripped_of_tensors, tx_list

    def deserialize(self, stubbed_obj, tensors: List[torch.Tensor]):
        def replace_tensor(stub, tensor_list):
            return tensor_list[stub.tensor_index]

        with self.catch_and_raise_for_large_object(stubbed_obj):
            return self._traverse_object(stubbed_obj, tensors, replace_tensor)

    def extract_stubs(self, stubbed_obj):
        def add_stub_to_list(obj, stub_list):
            stub_list.append(obj)
            return obj

        stub_list = []
        with self.catch_and_raise_for_large_object(stubbed_obj):
            self._traverse_object(stubbed_obj, stub_list, add_stub_to_list)
        return stub_list

    def _traverse_object(self, obj, tensor_list, callback):
        if isinstance(obj, TensorMeta):
            return callback(obj, tensor_list)

        if isinstance(obj, (bool, str, bytes, bytearray, int, float, Enum, type(None))):
            return obj

        if isinstance(obj, (list, tuple, set)) and not isinstance(obj, torch.Size):
            list_like_obj = []
            for item in obj:
                res = self._traverse_object(item, tensor_list, callback)
                list_like_obj.append(res)
            if is_instance_namedtuple(obj):
                # handling namedtuples
                cast_out = obj.__class__(*list_like_obj)
            else:
                cast_out = obj.__class__(list_like_obj)
            return cast_out

        if isinstance(obj, dict):
            # Recreate the instance based on
            # type of obj, and insert keys in the
            # order present in obj
            # Works only for mutable dicts
            instance_type = type(obj)
            if instance_type == collections.defaultdict:
                d = collections.defaultdict(obj.default_factory)
            else:
                d = instance_type()

            # Iteration order is deterministic on/after python3.7 / cpython3.6
            # https://docs.python.org/2/library/stdtypes.html#dict.items
            for sk, sv in obj.items():
                key = self._traverse_object(sk, tensor_list, callback)
                value = self._traverse_object(sv, tensor_list, callback)
                d[key] = value
            return d

        return obj

    def _replace_tensors_with_stubs(
        self,
        obj,
        memo: Dict,
        tx_list: List[torch.Tensor],
        stub_list: List[TensorMeta],
    ):
        if id(obj) in memo:
            return memo[id(obj)], False

        if isinstance(obj, (bool, str, bytes, bytearray, int, float, Enum, type(None))):
            return obj, False

        if isinstance(obj, (list, tuple, set)) and not isinstance(obj, torch.Size):
            list_like_obj = []
            seen_class_type = False
            for item in obj:
                res, ret_seen_cls_type = self._replace_tensors_with_stubs(item, memo, tx_list, stub_list)
                seen_class_type = seen_class_type or ret_seen_cls_type
                list_like_obj.append(res)
                memo[id(item)] = res
            if is_instance_namedtuple(obj):
                # handling namedtuples
                cast_out = obj.__class__(*list_like_obj)
            else:
                cast_out = obj.__class__(list_like_obj)
            memo[id(obj)] = cast_out
            return cast_out, seen_class_type

        if isinstance(obj, dict):
            # Obtain instance type from object and create an instance
            # of the instance type.
            # Insert keys in the order it is present in the obj
            # This approach only works for mutable dicts
            instance_type = type(obj)
            if instance_type == collections.defaultdict:
                d = collections.defaultdict(obj.default_factory)
            else:
                d = instance_type()
            seen_class_type = False
            for k, v in obj.items():
                stub_key, ret_seen_cls_type_key = self._replace_tensors_with_stubs(k, memo, tx_list, stub_list)
                memo[id(k)] = stub_key
                stub_value, ret_seen_cls_type_value = self._replace_tensors_with_stubs(v, memo, tx_list, stub_list)
                seen_class_type = seen_class_type or ret_seen_cls_type_key or ret_seen_cls_type_value
                memo[id(v)] = stub_value
                d[stub_key] = stub_value
            memo[id(obj)] = d
            return d, seen_class_type

        if isinstance(obj, torch.Tensor):
            stub = TensorMeta(
                len(tx_list),
                obj.dtype,
                obj.size(),
                obj.requires_grad,
                obj.device,
            )
            tx_list.append(obj)
            stub_list.append(stub)
            memo[id(obj)] = stub
            return stub, False

        # Unsupported class type, directly return
        memo[id(obj)] = obj
        return obj, False
