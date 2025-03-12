"""Tests for schedulers."""

import pytest
import warnings
import numpy as np
import torch
from torch.optim import SGD
from yoke.lr_schedulers import (
    power_decay_lr_calc,
    cos_decay_lr_calc,
    warmup_to_constant_lr,
    WarmUpPowerDecayScheduler,
    CosineWithWarmupScheduler,
    ConstantWithWarmupScheduler,
)


@pytest.fixture
def optimizer() -> SGD:
    """Fixture for creating a simple optimizer."""
    model = torch.nn.Linear(10, 1)
    return SGD(model.parameters(), lr=0.1)


def test_power_decay_lr_calc() -> None:
    """Test power_decay_lr_calc function."""
    step = 500
    anchor_lr = 0.001
    decay_power = 0.5
    warmup_steps = 1000
    dim_embed = 128

    lr = power_decay_lr_calc(
        step=step,
        anchor_lr=anchor_lr,
        decay_power=decay_power,
        warmup_steps=warmup_steps,
        dim_embed=dim_embed,
    )

    assert isinstance(lr, float)
    assert lr > 0
    assert lr <= anchor_lr


def test_cos_decay_lr_calc() -> None:
    """Test cos_decay_lr_calc function."""
    step = 700
    anchor_lr = 0.001
    warmup_steps = 100
    terminal_steps = 1000
    min_fraction = 0.1
    num_cycles = 0.5

    lr = cos_decay_lr_calc(
        step=step,
        anchor_lr=anchor_lr,
        warmup_steps=warmup_steps,
        terminal_steps=terminal_steps,
        min_fraction=min_fraction,
        num_cycles=num_cycles,
    )

    assert isinstance(lr, float)
    assert lr >= anchor_lr * min_fraction
    assert lr <= anchor_lr


def test_warmup_to_constant_lr() -> None:
    """Test warmup_to_constant_lr function."""
    step = 50
    warmup_steps = 100
    lr_constant = 0.001

    lr = warmup_to_constant_lr(
        step=step,
        warmup_steps=warmup_steps,
        lr_constant=lr_constant,
    )

    assert isinstance(lr, float)
    assert lr >= 0
    assert lr <= lr_constant


def test_warm_up_power_decay_scheduler(optimizer: SGD) -> None:
    """Test WarmUpPowerDecayScheduler."""
    scheduler = WarmUpPowerDecayScheduler(
        optimizer=optimizer,
        anchor_lr=0.001,
        decay_power=0.5,
        warmup_steps=100,
        dim_embed=4,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for step in range(1000):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            assert lr <= 0.001 / np.power(4, 0.5)


def test_cosine_with_warmup_scheduler(optimizer: SGD) -> None:
    """Test CosineWithWarmupScheduler."""
    scheduler = CosineWithWarmupScheduler(
        optimizer=optimizer,
        anchor_lr=0.001,
        terminal_steps=1000,
        warmup_steps=100,
        num_cycles=0.5,
        min_fraction=0.1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for step in range(1000):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            if step < 100:
                assert lr <= 0.001
            else:
                assert lr >= 0.001 * 0.1


def test_constant_with_warmup_scheduler(optimizer: SGD) -> None:
    """Test ConstantWithWarmupScheduler."""
    scheduler = ConstantWithWarmupScheduler(
        optimizer=optimizer,
        lr_constant=0.001,
        warmup_steps=100,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for step in range(200):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            assert lr > 0
            if step < 100:
                assert lr <= 0.001
            else:
                assert lr == 0.001
