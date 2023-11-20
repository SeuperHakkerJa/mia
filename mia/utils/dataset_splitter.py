import os
import os.path as osp
from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from mia import DATASET_DIR


def split_dataset(
    npz_file_name: str,
    n_data: int,
    test_size: float = 0.2,
    random_seed: int = None,
    save_indices: bool = False,
    load_indices_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a dataset contained in an npz file into train and test sets.

    Parameters:
    npz_file_name (str): Path to the .npz file.
    n_data (int): Number of data points to randomly draw from the npz file.
    test_size (float): Proportion of the dataset to include in the test split.
    random_seed (int): Seed for the random number generator for reproducibility.
    save_indices (bool): If True, saves the indices of the selected data to a file.
    load_indices_path (Optional[str]): Path to a file to load indices from instead of generating new ones.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (X_train, X_test, y_train, y_test)
    """

    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Load the npz file
    data = np.load(osp.join(DATASET_DIR, npz_file_name))

    # Extract 'x' and 'y' data
    X = data["x"]
    y = data["y"]

    # convert texas hospital label range from 1-100 to 0-99
    if npz_file_name == "texas.npz":
        y -= 1

    # Check if n_data is larger than the dataset size
    if n_data > X.shape[0]:
        raise ValueError("n_data is larger than the available dataset size.")

    if load_indices_path and os.path.exists(load_indices_path):
        # Load indices from file
        indices = np.load(load_indices_path)
    else:
        # Randomly select n_data samples
        indices = np.random.choice(X.shape[0], n_data, replace=False)

        if save_indices:
            # Check if the logs directory exists, if not, create it
            logs_dir = os.path.join(DATASET_DIR, "logs")
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            # Save the indices
            npz_file_basename = npz_file_name.split(".")[0]
            indices_file_path = f'selected_indices_{npz_file_basename.replace("/", "_")}_{random_seed}.npz'
            np.savez(indices_file_path, indices=indices)

    X_selected = X[indices]
    y_selected = y[indices]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_selected, test_size=test_size, random_state=random_seed
    )

    # Print basic information about the datasets
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(
        f"y_train range: {y_train.min()} - {y_train.max()}, y_test range: {y_test.min()} - {y_test.max()}"
    )

    return X_train, X_test, y_train, y_test
