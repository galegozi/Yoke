"""Pytest module to test the functions of the MLP class in yoke.models.vit.swin.encoder.

This module contains tests for the initialization and forward pass of the
MLP class, ensuring its components and behavior work as expected.
"""

import pytest
import torch
from torch import nn

from yoke.models.vit.swin.encoder import MLP


@pytest.fixture
def setup_mlp() -> tuple[MLP, int, int, int, torch.Tensor]:
    """Fixture to initialize an MLP instance and related parameters.

    Returns:
        A tuple containing the MLP instance, embedding size, batch size,
        sequence length, and input tensor.
    """
    emb_size = 64  # Default embedding size
    batch_size = 16
    seq_length = 10
    input_tensor = torch.randn(batch_size, seq_length, emb_size)
    return MLP(emb_size=emb_size), emb_size, batch_size, seq_length, input_tensor


def test_initialization(setup_mlp: tuple[MLP, int, int, int, torch.Tensor]) -> None:
    """Test if the MLP initializes correctly."""
    mlp, emb_size, _, _, _ = setup_mlp
    assert isinstance(mlp, nn.Module)
    assert hasattr(mlp, "ff")
    assert isinstance(mlp.ff, nn.Sequential)
    assert len(mlp.ff) == 3
    assert isinstance(mlp.ff[0], nn.Linear)
    assert mlp.ff[0].in_features == emb_size
    assert mlp.ff[0].out_features == 4 * emb_size
    assert isinstance(mlp.ff[1], nn.GELU)
    assert isinstance(mlp.ff[2], nn.Linear)
    assert mlp.ff[2].in_features == 4 * emb_size
    assert mlp.ff[2].out_features == emb_size


def test_forward_pass(setup_mlp: tuple[MLP, int, int, int, torch.Tensor]) -> None:
    """Test the forward pass outputs the correct shape."""
    mlp, _, batch_size, seq_length, input_tensor = setup_mlp
    output = mlp(input_tensor)
    assert output.shape == (batch_size, seq_length, mlp.ff[2].out_features)


@pytest.mark.parametrize("emb_size", [32, 64, 128])
def test_forward_pass_with_different_embedding_size(emb_size: int) -> None:
    """Test the MLP with different embedding sizes."""
    batch_size = 16
    seq_length = 10
    mlp = MLP(emb_size=emb_size)
    input_tensor = torch.randn(batch_size, seq_length, emb_size)
    output = mlp(input_tensor)
    assert output.shape == (batch_size, seq_length, emb_size)
