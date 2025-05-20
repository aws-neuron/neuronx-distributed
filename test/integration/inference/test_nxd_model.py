from typing import Dict, List, Optional

import pytest
import torch

from neuronx_distributed.trace.nxd_model import NxDModel

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_add_basic():
    nxd_model = NxDModel(32)
    nxd_model.add(
        None,
        None,
        None,
        None,
        None,
        None,
        None
    )
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_add_multiple():
    nxd_model = NxDModel(32)
    nxd_model.add(
        None,
        None,
        None,
        None,
        None,
        None,
        None
    )
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_set_weights_on_cpu():
    nxd_model = NxDModel(32)
    nxd_model.set_weights(
        None,
    )
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_set_weights_on_hbm():
    nxd_model = NxDModel(32)
    nxd_model.set_weights(
        None,
    )
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_set_weights_layout_transformation_applied():
    nxd_model = NxDModel(32)
    nxd_model.set_weights(
        None,
    )
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_to_neuron():
    nxd_model = NxDModel(32)
    nxd_model.to_neuron()
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_convert_dict_to_ordered_list_basic():
    nxd_model = NxDModel(32)
    nxd_model.convert_dict_to_ordered_list(
        None
    )
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_convert_dict_to_ordered_list_with_pos_args():
    nxd_model = NxDModel(32)
    nxd_model.convert_dict_to_ordered_list(
        None
    )
    # TODO: further develop test


@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_convert_dict_to_ordered_list_skip_kwargs():
    nxd_model = NxDModel(32)
    nxd_model.convert_dict_to_ordered_list(
        None
    )
    # TODO: further develop test


@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_convert_dict_to_ordered_list_skip_kwargs_and_with_pos_args():
    nxd_model = NxDModel(32)
    nxd_model.convert_dict_to_ordered_list(
        None
    )
    # TODO: further develop test


@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_convert_dict_to_ordered_list_invalid_pos_args():
    nxd_model = NxDModel(32)
    nxd_model.convert_dict_to_ordered_list(
        None
    )
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_convert_dict_to_ordered_list_invalid_unrecognized_kwarg():
    nxd_model = NxDModel(32)
    nxd_model.convert_dict_to_ordered_list(
        None
    )
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_router_basic():
    nxd_model = NxDModel(32)
    nxd_model.router(None)
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_router_no_route_available():
    nxd_model = NxDModel(32)
    nxd_model.router(None)
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_forward_basic():
    nxd_model = NxDModel(32)
    nxd_model(None)
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_forward_kwargs():
    nxd_model = NxDModel(32)
    nxd_model(None)
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_forward_pos_and_kwargs():
    nxd_model = NxDModel(32)
    nxd_model(None)
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_forward_specific_model_name():
    nxd_model = NxDModel(32)
    nxd_model(None)
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_save_basic():
    nxd_model = NxDModel(32)
    nxd_model.save(None)
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_save_multi_bucket():
    nxd_model = NxDModel(32)
    nxd_model.save(None)
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_save_with_weights():
    nxd_model = NxDModel(32)
    nxd_model.save(None)
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_save_with_kwargs():
    nxd_model = NxDModel(32)
    nxd_model.save(None)
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_load():
    NxDModel.load(None)
    # TODO: further develop test

@pytest.mark.xfail(reason="To Be Implemented", raises=NotImplementedError)
def test_load_with_weights():
    NxDModel.load(None)
    # TODO: further develop test
