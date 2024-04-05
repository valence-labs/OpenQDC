"""Path hack to make tests work."""

import numpy as np
import pytest

from openqdc.datasets.potential.dummy import Dummy  # noqa: E402
from openqdc.utils.package_utils import has_package

if has_package("torch"):
    import torch

if has_package("jax"):
    import jax

format_to_type = {
    "numpy": np.ndarray,
    "torch": torch.Tensor if has_package("torch") else None,
    "jax": jax.numpy.ndarray if has_package("jax") else None,
}


@pytest.fixture
def ds():
    return Dummy()


def test_dummy(ds):
    assert ds is not None
    assert len(ds) == 9999
    assert ds[100]


@pytest.mark.parametrize("format", ["numpy", "torch", "jax"])
def test_array_format(format):
    if not has_package(format):
        pytest.skip(f"{format} is not installed, skipping test")

    ds = Dummy(array_format=format)

    keys = [
        "positions",
        "atomic_numbers",
        "charges",
        "energies",
        "forces",
        "e0",
        "formation_energies",
        "per_atom_formation_energies",
    ]

    data = ds[0]
    for key in keys:
        assert isinstance(data[key], format_to_type[format])


def test_transform():
    def custom_fn(bunch):
        # create new name
        bunch.new_key = bunch.name + bunch.subset
        return bunch

    ds = Dummy(transform=custom_fn)

    data = ds[0]

    assert "new_key" in data
    assert data["new_key"] == data["name"] + data["subset"]
