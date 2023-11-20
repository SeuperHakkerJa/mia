import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm


class ShadowModelBundle(BaseEstimator):
    """
    A bundle of shadow models for training and transforming data for model attack.

    Parameters:
    - model_fn: Function to build a new shadow model.
    - shadow_dataset_size: Size of training data for each shadow model.
    - num_models: Number of shadow models (default=20).
    - seed: Random seed (default=42).
    - serializer: Serializer for models, stored in memory if None (default=None).
    """

    MODEL_ID_FMT = "shadow_%d"

    def __init__(
        self, model_fn, shadow_dataset_size, num_models=20, seed=42, serializer=None
    ):
        super().__init__()
        self.model_fn = model_fn
        self.shadow_dataset_size = shadow_dataset_size
        self.num_models = num_models
        self.seed = seed
        self.serializer = serializer
        self._reset_random_state()

    def train_and_transform(self, X, y, verbose=False, fit_kwargs=None):
        """
        Trains shadow models and generates a dataset for the attack model.

        Parameters:
        - X, y: Training data and labels.
        - verbose: Display progress (default=False).
        - fit_kwargs: Additional arguments for model fitting (default=None).
        """
        self._train_shadow_models(X, y, verbose, fit_kwargs)
        return self._generate_attack_data(verbose)

    def _reset_random_state(self):
        self._prng = np.random.RandomState(self.seed)

    def _train_shadow_models(self, X, y, verbose=False, pseudo=False, fit_kwargs=None):
        """
        Internal method to train shadow models.

        Parameters:
        - X, y: Training data and labels.
        - verbose: Display progress (default=False).
        - pseudo: If True, models are not actually fitted (default=False).
        - fit_kwargs: Additional arguments for model fitting (default=None).
        """
        self.shadow_train_indices_ = []
        self.shadow_test_indices_ = []
        self.shadow_models_ = [] if self.serializer is None else None

        fit_kwargs = fit_kwargs or {}
        indices = np.arange(X.shape[0])

        for i in self._iterate_models(verbose):
            shadow_indices = self._prng.choice(
                indices, 2 * self.shadow_dataset_size, replace=False
            )
            train_indices, test_indices = np.split(shadow_indices, 2)
            self.shadow_train_indices_.append(train_indices)
            self.shadow_test_indices_.append(test_indices)

            if not pseudo:
                self._fit_single_model(
                    i, X[train_indices], y[train_indices], fit_kwargs
                )

        self.X_fit_, self.y_fit_ = X, y
        self._reset_random_state()

    def _fit_single_model(self, model_index, X_train, y_train, fit_kwargs):
        """
        Fits a single shadow model.

        Parameters:
        - model_index: Index of the shadow model.
        - X_train, y_train: Training data and labels.
        - fit_kwargs: Additional arguments for model fitting.
        """
        shadow_model = self.model_fn()
        shadow_model.fit(X_train, y_train, **fit_kwargs)
        model_id = ShadowModelBundle.MODEL_ID_FMT % model_index
        self.serializer.save(
            model_id, shadow_model
        ) if self.serializer else self.shadow_models_.append(shadow_model)

    def _iterate_models(self, verbose=False, indices=None):
        """
        Iterates over model indices, optionally with a progress bar.

        Parameters:
        - verbose: Display progress (default=False).
        - indices: Specific model indices to iterate over (default=None).
        """
        indices = range(self.num_models) if indices is None else indices
        return tqdm(indices) if verbose else indices

    def _generate_attack_data(self, verbose=False):
        """
        Generates data for training the attack model.

        Parameters:
        - verbose: Display progress (default=False).
        """
        shadow_data, shadow_labels = [], []

        for i in self._iterate_models(verbose):
            shadow_model = self._load_model(i)
            train_data = (
                self.X_fit_[self.shadow_train_indices_[i]],
                self.y_fit_[self.shadow_train_indices_[i]],
            )
            test_data = (
                self.X_fit_[self.shadow_test_indices_[i]],
                self.y_fit_[self.shadow_test_indices_[i]],
            )
            data, labels = prepare_attack_data(shadow_model, train_data, test_data)

            shadow_data.append(data)
            shadow_labels.append(labels)

        X_transformed = np.vstack(shadow_data).astype("float32")
        y_transformed = np.hstack(shadow_labels).astype("float32")
        return X_transformed, y_transformed

    def _load_model(self, model_index):
        """
        Loads a shadow model by index.

        Parameters:
        - model_index: Index of the shadow model.
        """
        if self.serializer is not None:
            model_id = ShadowModelBundle.MODEL_ID_FMT % model_index
            return self.serializer.load(model_id)
        else:
            return self.shadow_models_[model_index]
