import os
import os.path as osp

import numpy as np
import pytest

from mia import DATASET_DIR
from mia.utils import split_dataset


def create_dummy_dataset(file_name: str, size: int = 1000):
    """
    Create a dummy npz file with random data for testing.

    Parameters:
    file_name (str): The name of the file to be created.
    size (int): The number of data points to create.
    """
    X = np.random.rand(size, 10)  # Dummy features
    y = np.random.randint(0, 2, size)  # Dummy binary labels

    np.savez(file_name, x=X, y=y)


@pytest.fixture(scope="module")
def dummy_dataset():
    """
    Pytest fixture to create and delete a dummy dataset for testing.
    """
    file_name = "dummy.npz"
    create_dummy_dataset(osp.join(DATASET_DIR, file_name))
    yield file_name
    os.remove(osp.join(DATASET_DIR, file_name))


def test_split_dataset_dimensions(dummy_dataset):
    """
    Test if the split_dataset function correctly splits the dataset into
    the expected dimensions.
    """
    n_data = 200
    test_size = 0.25
    X_train, X_test, y_train, y_test = split_dataset(
        dummy_dataset, n_data, test_size=test_size
    )

    assert (
        X_train.shape[0] == n_data * (1 - test_size)
        and X_test.shape[0] == n_data * test_size
    )
    assert (
        y_train.shape[0] == n_data * (1 - test_size)
        and y_test.shape[0] == n_data * test_size
    )


def test_split_dataset_reproducibility(dummy_dataset):
    """
    Test if the split_dataset function is reproducible with a given random seed.
    """
    n_data = 200
    random_seed = 42
    X_train_1, X_test_1, y_train_1, y_test_1 = split_dataset(
        dummy_dataset, n_data, random_seed=random_seed
    )
    X_train_2, X_test_2, y_train_2, y_test_2 = split_dataset(
        dummy_dataset, n_data, random_seed=random_seed
    )

    assert np.array_equal(X_train_1, X_train_2) and np.array_equal(X_test_1, X_test_2)
    assert np.array_equal(y_train_1, y_train_2) and np.array_equal(y_test_1, y_test_2)


# Note: More tests can be added to cover other aspects like error handling, loading from saved indices, etc.
