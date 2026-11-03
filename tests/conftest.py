"""Shared fixtures."""

import pytest

from heart_disease.data import load_dataset


@pytest.fixture(scope="session")
def dataset():
    """The cleaned real dataset."""
    return load_dataset()
