"""Test surrogate CNN."""

import pytest
import torch
from torch import nn

from yoke.models.surrogateCNNmodules import tCNNsurrogate


@pytest.fixture
def default_model() -> tCNNsurrogate:
    """Fixture for default tCNNsurrogate instance."""
    return tCNNsurrogate()


def test_initialization(default_model: tCNNsurrogate) -> None:
    """Test initialization of the model."""
    assert isinstance(default_model, tCNNsurrogate)
    assert isinstance(default_model.dense_expand, nn.Linear)
    assert isinstance(default_model.initTConv, nn.ConvTranspose2d)
    assert isinstance(default_model.CompoundConvTList, nn.ModuleList)
    assert isinstance(default_model.final_tconv, nn.ConvTranspose2d)


def test_forward_output_shape(default_model: tCNNsurrogate) -> None:
    """Test forward method output shape."""
    batch_size = 4
    input_tensor = torch.rand(batch_size, default_model.input_size)
    output = default_model(input_tensor)
    expected_shape = (
        batch_size,
        default_model.output_image_channels,
        *default_model.output_image_size,
    )
    assert output.shape == expected_shape


def test_forward_pass(default_model: tCNNsurrogate) -> None:
    """Test forward pass of the model."""
    batch_size = 4
    input_tensor = torch.rand(batch_size, default_model.input_size)
    output = default_model(input_tensor)
    assert torch.is_tensor(output)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.parametrize(
    "input_size,linear_features,kernel,nfeature_list,"
    "output_image_size,output_image_channels",
    [
        (30, (7, 5, 128), (3, 3), [128, 64, 32], (224, 224), 3),
        (16, (4, 4, 64), (5, 5), [64, 64, 32], (128, 128), 1),
    ],
)
def test_custom_initialization(
    input_size: int,
    linear_features: tuple[int, int],
    kernel: tuple[int, int],
    nfeature_list: list[int],
    output_image_size: tuple[int, int],
    output_image_channels: int,
) -> None:
    """Test model initialization with custom parameters."""
    model = tCNNsurrogate(
        input_size=input_size,
        linear_features=linear_features,
        kernel=kernel,
        nfeature_list=nfeature_list,
        output_image_size=output_image_size,
        output_image_channels=output_image_channels,
    )
    assert model.input_size == input_size
    assert model.linear_features == linear_features
    assert model.kernel == kernel
    assert model.nfeature_list == nfeature_list
    assert model.output_image_size == output_image_size
    assert model.output_image_channels == output_image_channels


def test_model_gradients(default_model: tCNNsurrogate) -> None:
    """Test gradients during backpropagation."""
    batch_size = 2
    input_tensor = torch.rand(batch_size, default_model.input_size, requires_grad=True)
    output = default_model(input_tensor)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None
    assert not torch.isnan(input_tensor.grad).any()
