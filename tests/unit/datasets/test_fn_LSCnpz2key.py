"""Unit tests for the LSCnpz2key function in the yoke.datasets.lsc_dataset module.

These tests ensure that the function correctly extracts keys from .npz filenames,
whether they include directory paths or not.
"""

import pytest
from yoke.datasets.lsc_dataset import LSCnpz2key


@pytest.mark.parametrize(
    "npz_file, exp_key",
    [
        (
            "/lustre/vescratch1/exempt/artimis/mpmm/lsc240420/lsc240420_id00523_pvi_idx00025.npz",
            "lsc240420_id00523",
        ),
        ("lsc240420_id00523_pvi_idx00025.npz", "lsc240420_id00523"),
        (
            "/lustre/vescratch1/exempt/artimis/mpmm/lsc240420/lsc240420_id00960_pvi_idx00012.npz",
            "lsc240420_id00960",
        ),
        ("lsc240420_id00960_pvi_idx00012.npz", "lsc240420_id00960"),
        (
            "/lustre/vescratch1/exempt/artimis/mpmm/lsc240420/lsc240420_id00310_pvi_idx00086.npz",
            "lsc240420_id00310",
        ),
        ("lsc240420_id00310_pvi_idx00086.npz", "lsc240420_id00310"),
    ],
)
def test_LSCnpz2key(npz_file: str, exp_key: str) -> None:
    """Test the LSCnpz2key function.

    Verifies that the function correctly extracts keys from .npz filenames,
    with or without directory paths.

    Args:
        npz_file (str): The input .npz filename.
        exp_key (str): The expected key extracted from the filename.

    Raises:
        AssertionError: If the result does not match the expected key.
    """
    result = LSCnpz2key(npz_file)
    assert result == exp_key, f"For {npz_file}, expected {exp_key} but got {result}"
