"""Tests for attention sink functionality in windowed MSA classes."""

import pytest
import torch
from yoke.models.vit.swin.windowed_msa import (
    WindowMSA, ShiftedWindowMSA, WindowCosMSA, ShiftedWindowCosMSA
)


@pytest.fixture
def device() -> str:
    """Get the device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def test_input(device: str) -> torch.Tensor:
    """Create a test input tensor."""
    return torch.rand(2, 56 * 40, 64).to(device)


@pytest.fixture
def msa_params() -> dict:
    """Common parameters for MSA classes."""
    return {
        "emb_size": 64,
        "num_heads": 8,
        "patch_grid_size": (56, 40),
        "window_size": (8, 10),
    }


class TestAttentionSink:
    """Test class for attention sink functionality."""

    @pytest.mark.parametrize("cls", [
        WindowMSA, ShiftedWindowMSA, WindowCosMSA, ShiftedWindowCosMSA
    ])
    def test_attention_sink_enabled(self, cls, msa_params: dict, test_input: torch.Tensor, device: str) -> None:
        """Test that attention sink works when enabled."""
        model = cls(**msa_params, use_attention_sink=True).to(device)
        
        # Test forward pass
        output = model(test_input)
        assert output.shape == test_input.shape
        
        # Test parameter existence and shape
        assert hasattr(model, 'attn_sink')
        assert model.attn_sink is not None
        
        expected_shape = (1, msa_params["num_heads"], 1, 1, 1, 
                         msa_params["window_size"][0] * msa_params["window_size"][1])
        assert model.attn_sink.shape == expected_shape
        assert model.attn_sink.requires_grad

    @pytest.mark.parametrize("cls", [
        WindowMSA, ShiftedWindowMSA, WindowCosMSA, ShiftedWindowCosMSA
    ])
    def test_attention_sink_disabled(self, cls, msa_params: dict, test_input: torch.Tensor, device: str) -> None:
        """Test that attention sink is None when disabled."""
        model = cls(**msa_params, use_attention_sink=False).to(device)
        
        # Test forward pass
        output = model(test_input)
        assert output.shape == test_input.shape
        
        # Test parameter is None
        assert hasattr(model, 'attn_sink')
        assert model.attn_sink is None

    @pytest.mark.parametrize("cls", [
        WindowMSA, ShiftedWindowMSA, WindowCosMSA, ShiftedWindowCosMSA
    ])
    def test_attention_sink_default_enabled(self, cls, msa_params: dict, device: str) -> None:
        """Test that attention sink is enabled by default."""
        model = cls(**msa_params).to(device)
        
        assert hasattr(model, 'use_attention_sink')
        assert model.use_attention_sink is True
        assert model.attn_sink is not None

    def test_gradient_flow(self, msa_params: dict, device: str) -> None:
        """Test that gradients flow to the attention sink parameter."""
        model = WindowMSA(**msa_params, use_attention_sink=True).to(device)
        x = torch.rand(1, 56 * 40, 64, requires_grad=True).to(device)
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, torch.zeros_like(output))
        loss.backward()
        
        # Check that attn_sink has gradients
        assert model.attn_sink.grad is not None
        assert not torch.allclose(model.attn_sink.grad, torch.zeros_like(model.attn_sink.grad))

    def test_backward_compatibility(self, msa_params: dict, test_input: torch.Tensor, device: str) -> None:
        """Test that models without use_attention_sink argument still work."""
        # Create models without specifying use_attention_sink (should default to True)
        models = [
            WindowMSA(**msa_params).to(device),
            ShiftedWindowMSA(**msa_params).to(device),
            WindowCosMSA(**msa_params).to(device),
            ShiftedWindowCosMSA(**msa_params).to(device),
        ]
        
        for model in models:
            output = model(test_input)
            assert output.shape == test_input.shape
            assert model.use_attention_sink is True
            assert model.attn_sink is not None

    def test_state_dict_behavior(self, msa_params: dict, device: str) -> None:
        """Test that attn_sink appears in state dict when enabled, not when disabled."""
        # With attention sink enabled
        model_enabled = WindowMSA(**msa_params, use_attention_sink=True).to(device)
        state_dict_enabled = model_enabled.state_dict()
        assert 'attn_sink' in state_dict_enabled
        
        # With attention sink disabled
        model_disabled = WindowMSA(**msa_params, use_attention_sink=False).to(device)
        state_dict_disabled = model_disabled.state_dict()
        assert 'attn_sink' not in state_dict_disabled