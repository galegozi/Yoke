"""Tests for torch training utilities."""

import os
import pytest
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tempfile import TemporaryDirectory
import h5py
from collections.abc import Generator
from yoke.torch_training_utils import count_torch_params
from yoke.torch_training_utils import save_model_and_optimizer_hdf5
from yoke.torch_training_utils import load_model_and_optimizer_hdf5
from yoke.torch_training_utils import make_dataloader


class SimpleModel(nn.Module):
    """A simple test model for parameter counting."""

    def __init__(self) -> None:
        """Setup the model."""
        super().__init__()
        self.fc1 = nn.Linear(10, 20)  # 10*20 + 20 = 220 parameters
        self.fc2 = nn.Linear(20, 5)  # 20*5 + 5 = 105 parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward method for module."""
        return self.fc2(self.fc1(x))


class SimpleModel2(nn.Module):
    """A simple test model."""

    def __init__(self) -> None:
        """Setup the model."""
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward method for module."""
        return self.fc(x)


@pytest.fixture
def simple_model() -> Generator[SimpleModel, None, None]:
    """Fixture for creating a simple model."""
    model = SimpleModel()
    yield model


@pytest.fixture
def simple_model_and_optimizer() -> tuple[nn.Module, optim.Optimizer]:
    """Fixture to create a simple model and optimizer."""
    model = SimpleModel2()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return model, optimizer


def test_count_torch_params_trainable(simple_model: SimpleModel) -> None:
    """Test count_torch_params for trainable parameters."""
    # By default, all parameters are trainable
    assert count_torch_params(simple_model) == 325  # Total parameters = 220 + 105


def test_count_torch_params_non_trainable(simple_model: SimpleModel) -> None:
    """Test count_torch_params for non-trainable parameters."""
    # Freeze all parameters
    for param in simple_model.parameters():
        param.requires_grad = False
    assert count_torch_params(simple_model) == 0  # No trainable parameters
    assert count_torch_params(simple_model, trainable=False) == 325  # Total parameters


def test_count_torch_params_partial_trainable(simple_model: SimpleModel) -> None:
    """Test count_torch_params when only some parameters are trainable."""
    # Freeze only fc2 layer
    for param in simple_model.fc2.parameters():
        param.requires_grad = False
    assert count_torch_params(simple_model) == 220  # Only fc1 parameters are trainable
    assert count_torch_params(simple_model, trainable=False) == 325  # Total parameters


def test_save_model_and_optimizer_hdf5(
    simple_model_and_optimizer: tuple[nn.Module, optim.Optimizer],
) -> None:
    """Test saving a model and optimizer to an HDF5 file."""
    model, optimizer = simple_model_and_optimizer
    epoch = 5

    with TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "checkpoint.h5")

        # Save the model and optimizer state
        save_model_and_optimizer_hdf5(model, optimizer, epoch, filepath)

        # Validate the saved file
        with h5py.File(filepath, "r") as h5f:
            # Check epoch attribute
            assert h5f.attrs["epoch"] == epoch

            # Check model parameters
            for name, param in model.named_parameters():
                if param.data.ndimension() == 0:  # Scalar parameter
                    saved_attrs = h5f.attrs[f"model/parameters/{name}"]
                    assert saved_attrs == pytest.approx(param.item())
                else:
                    saved_param = h5f[f"model/parameters/{name}"][:]
                    assert torch.equal(torch.tensor(saved_param), param.detach().cpu())

            # Check model buffers
            for name, buffer in model.named_buffers():
                if buffer.ndimension() == 0:  # Scalar buffer
                    saved_attrs = h5f.attrs[f"model/buffers/{name}"]
                    assert saved_attrs == pytest.approx(buffer.item())
                else:
                    saved_buffer = h5f[f"model/buffers/{name}"][:]
                    assert torch.equal(torch.tensor(saved_buffer), buffer.cpu())

            # Check optimizer state_dict param_groups
            optimizer_state = optimizer.state_dict()
            for idx, group in enumerate(optimizer_state["param_groups"]):
                group_name = f"optimizer/group{idx}"
                for k, v in group.items():
                    if isinstance(v, (int, float)):
                        assert h5f.attrs[f"{group_name}/{k}"] == v
                    elif isinstance(v, list):
                        saved_list = h5f[f"{group_name}/{k}"][:]
                        assert saved_list.tolist() == v

            # Check optimizer state_dict states
            for idx, (state_id, state) in enumerate(optimizer_state["state"].items()):
                state_name = f"optimizer/state{idx}"
                for k, v in state.items():
                    saved_state = h5f[f"{state_name}/{k}"][:]
                    assert torch.equal(torch.tensor(saved_state), v.cpu())


def test_save_model_and_optimizer_hdf5_compiled_model(
    simple_model_and_optimizer: tuple[nn.Module, optim.Optimizer],
) -> None:
    """Test saving a compiled model and optimizer."""
    model, optimizer = simple_model_and_optimizer
    epoch = 5

    # Mock compiled model for testing
    class MockCompiledModel(nn.Module):
        def __init__(self, original_model: nn.Module) -> None:
            super().__init__()
            self._orig_mod = original_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._orig_mod(x)

    compiled_model = MockCompiledModel(model)

    with TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "compiled_checkpoint.h5")

        # Save the compiled model and optimizer state
        save_model_and_optimizer_hdf5(
            compiled_model, optimizer, epoch, filepath, compiled=True
        )

        # Validate the saved file
        with h5py.File(filepath, "r") as h5f:
            # Check epoch attribute
            assert h5f.attrs["epoch"] == epoch
            # Check model parameters (same as above)
            for name, param in model.named_parameters():
                if param.data.ndimension() == 0:  # Scalar parameter
                    saved_attrs = h5f.attrs[f"model/parameters/{name}"]
                    assert saved_attrs == pytest.approx(param.item())
                else:
                    saved_param = h5f[f"model/parameters/{name}"][:]
                    assert torch.equal(torch.tensor(saved_param), param.detach().cpu())


def test_load_model_and_optimizer_hdf5(
    simple_model_and_optimizer: tuple[nn.Module, optim.Optimizer],
) -> None:
    """Test loading a model and optimizer from an HDF5 file."""
    model, optimizer = simple_model_and_optimizer
    epoch = 5

    with TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "checkpoint.h5")

        # Save the model and optimizer state
        save_model_and_optimizer_hdf5(model, optimizer, epoch, filepath)

        # Create new model and optimizer instances for loading
        loaded_model = SimpleModel2()
        loaded_optimizer = optim.SGD(loaded_model.parameters(), lr=0.01, momentum=0.9)

        # Load the state into the new instances
        loaded_epoch = load_model_and_optimizer_hdf5(
            loaded_model, loaded_optimizer, filepath
        )

        # Verify the epoch is correctly restored
        assert loaded_epoch == epoch

        # Verify the model parameters are correctly restored
        for original, loaded in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(original.cpu(), loaded.cpu())

        # Verify the optimizer state is correctly restored
        original_state = optimizer.state_dict()
        loaded_state = loaded_optimizer.state_dict()
        assert original_state.keys() == loaded_state.keys()
        for key in original_state:
            if isinstance(original_state[key], list):
                for orig, load in zip(original_state[key], loaded_state[key]):
                    assert orig == load
            else:
                assert original_state[key] == loaded_state[key]


class DummyDataset(Dataset):
    """A dummy dataset for testing."""

    def __init__(self, size: int = 1000) -> None:
        """Initialize dataset."""
        self.data = torch.arange(size)

    def __len__(self) -> int:
        """Datasets must have length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Dummy getitem."""
        return self.data[idx]


@pytest.fixture
def dummy_dataset() -> DummyDataset:
    """Fixture to create a dummy dataset."""
    return DummyDataset()


def test_make_dataloader(dummy_dataset: DummyDataset) -> None:
    """Test make_dataloader with default arguments."""
    batch_size = 8
    num_batches = 10

    dataloader = make_dataloader(
        dummy_dataset, batch_size=batch_size, num_batches=num_batches
    )

    # Check if the dataloader is a DataLoader instance
    assert isinstance(dataloader, DataLoader)

    # Validate the length of the dataloader
    assert len(dataloader) == num_batches

    # Validate the total number of samples
    total_samples = batch_size * num_batches
    sampled_data = list(dataloader)
    assert len(sampled_data) == num_batches
    assert sum(batch.shape[0] for batch in sampled_data) == total_samples


@pytest.mark.parametrize(
    "batch_size, num_batches, expected_length",
    [
        (4, 5, 5),  # Small batch size and number of batches
        (16, 10, 10),  # Larger batch size
        (32, 2, 2),  # Fewer batches with large batch size
    ],
)
def test_make_dataloader_parametrized(
    dummy_dataset: DummyDataset,
    batch_size: int,
    num_batches: int,
    expected_length: int,
) -> None:
    """Test make_dataloader with various batch sizes and number of batches."""
    dataloader = make_dataloader(
        dummy_dataset, batch_size=batch_size, num_batches=num_batches
    )
    assert len(dataloader) == expected_length

    # Validate all batches have the correct size except the last one if dataset
    # is too small
    sampled_data = list(dataloader)
    for batch in sampled_data[:-1]:
        assert batch.shape[0] == batch_size
