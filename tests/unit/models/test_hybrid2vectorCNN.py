"""Test hybrid CNN model."""

import pytest
import torch
from torch import nn

from yoke.models.hybridCNNmodules import hybrid2vectorCNN


@pytest.fixture
def default_hybrid2vector_cnn() -> hybrid2vectorCNN:
    """Fixture for default hybrid2vectorCNN instance."""
    return hybrid2vectorCNN()


def test_hybrid2vector_cnn_initialization(
    default_hybrid2vector_cnn: hybrid2vectorCNN,
) -> None:
    """Test initialization of hybrid2vectorCNN."""
    assert isinstance(default_hybrid2vector_cnn, hybrid2vectorCNN)
    assert isinstance(default_hybrid2vector_cnn.interpH1, nn.Module)
    assert isinstance(default_hybrid2vector_cnn.reduceH1, nn.Module)
    assert isinstance(default_hybrid2vector_cnn.lin_embed_h1, nn.Module)
    assert isinstance(default_hybrid2vector_cnn.final_mlp, nn.Module)


def test_hybrid2vector_cnn_forward_output_shape(
    default_hybrid2vector_cnn: hybrid2vectorCNN,
) -> None:
    """Test forward method output shape."""
    batch_size = 4
    vector_size = default_hybrid2vector_cnn.input_vector_size
    img_size = default_hybrid2vector_cnn.img_size
    y = torch.rand(batch_size, vector_size)
    h1 = torch.rand(batch_size, *img_size)
    h2 = torch.rand(batch_size, *img_size)
    output = default_hybrid2vector_cnn(y, h1, h2)
    assert output.shape == (batch_size, default_hybrid2vector_cnn.output_dim)


def test_hybrid2vector_cnn_forward_pass(
    default_hybrid2vector_cnn: hybrid2vectorCNN,
) -> None:
    """Test forward pass of hybrid2vectorCNN."""
    batch_size = 4
    vector_size = default_hybrid2vector_cnn.input_vector_size
    img_size = default_hybrid2vector_cnn.img_size
    y = torch.rand(batch_size, vector_size)
    h1 = torch.rand(batch_size, *img_size)
    h2 = torch.rand(batch_size, *img_size)
    output = default_hybrid2vector_cnn(y, h1, h2)
    assert torch.is_tensor(output)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.parametrize(
    "img_size,input_vector_size,output_dim,features",
    [
        ((1, 64, 64), 16, 2, 8),
        ((3, 128, 128), 32, 4, 16),
    ],
)
def test_hybrid2vector_cnn_custom_initialization(
    img_size: tuple[int, int, int],
    input_vector_size: int,
    output_dim: int,
    features: int,
) -> None:
    """Test hybrid2vectorCNN initialization with custom parameters."""
    model = hybrid2vectorCNN(
        img_size=img_size,
        input_vector_size=input_vector_size,
        output_dim=output_dim,
        features=features,
    )
    assert model.img_size == img_size
    assert model.input_vector_size == input_vector_size
    assert model.output_dim == output_dim
    assert model.features == features


def test_hybrid2vector_cnn_gradient_backpropagation(
    default_hybrid2vector_cnn: hybrid2vectorCNN,
) -> None:
    """Test gradients during backpropagation for hybrid2vectorCNN."""
    batch_size = 4
    vector_size = default_hybrid2vector_cnn.input_vector_size
    img_size = default_hybrid2vector_cnn.img_size

    # Create inputs with requires_grad=True to track gradients
    y = torch.rand(batch_size, vector_size, requires_grad=True)
    h1 = torch.rand(batch_size, *img_size, requires_grad=True)
    h2 = torch.rand(batch_size, *img_size, requires_grad=True)

    # Forward pass
    output = default_hybrid2vector_cnn(y, h1, h2)

    # Compute a simple loss and backpropagate
    loss = output.sum()
    loss.backward()

    # Check gradients for all inputs
    assert y.grad is not None, "Gradients for vector input are None."
    assert h1.grad is not None, "Gradients for first image input are None."
    assert h2.grad is not None, "Gradients for second image input are None."

    # Ensure gradients are finite
    assert not torch.isnan(y.grad).any(), "Gradients for vector input contain NaN."
    assert not torch.isnan(h1.grad).any(), "Gradients for 1st image input contain NaN."
    assert not torch.isnan(h2.grad).any(), "Gradients for 2nd image input contain NaN."
    assert not torch.isinf(y.grad).any(), "Gradients for vector input contain Inf."
    assert not torch.isinf(h1.grad).any(), "Gradients for 1st image input contain Inf."
    assert not torch.isinf(h2.grad).any(), "Gradients for 2nd image input contain Inf."
