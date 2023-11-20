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

import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

def prepare_attack_data(model, data_in, data_out):
    """
    Prepares data for the attack model in a suitable format.

    Parameters:
    - model: Classifier model.
    - data_in: Tuple (X, y) of data used for training.
    - data_out: Tuple (X, y) of data not used for training.

    Returns:
    Tuple (X, y) for the attack classifier.
    """
    X_in, y_in = data_in
    X_out, y_out = data_out
    y_hat_in = model.predict_proba(X_in)
    y_hat_out = model.predict_proba(X_out)

    labels = np.hstack([np.ones(y_in.shape[0]), np.zeros(y_out.shape[0])])
    data = np.vstack([np.c_[y_hat_in, y_in], np.c_[y_hat_out, y_out]])
    return data, labels

class AttackModelBundle(BaseEstimator):
    """
    A bundle of attack models for each class of the target model.

    Parameters:
    - model_fn: Function to build a new attack model.
    - num_classes: Number of classes in the target model.
    - serializer: Optional serializer for models (default=None).
    - class_one_hot_coded: Whether class labels are one-hot encoded (default=True).
    """

    MODEL_ID_FMT = "attack_%d"

    def __init__(self, model_fn, num_classes, serializer=None, class_one_hot_coded=True):
        self.model_fn = model_fn
        self.num_classes = num_classes
        self.serializer = serializer
        self.class_one_hot_coded = class_one_hot_coded

    def fit(self, X, y, verbose=False, fit_kwargs=None):
        """
        Trains the attack models.

        Parameters:
        - X, y: Data and labels from shadow models.
        - verbose: Display progress bar (default=False).
        - fit_kwargs: Additional arguments for model fitting (default=None).
        """
        X_classes = X[:, self.num_classes:]
        datasets_by_class = self._split_data_by_class(X, y, X_classes)

        self.attack_models_ = [] if self.serializer is None else None
        dataset_iter = tqdm(datasets_by_class) if verbose else datasets_by_class

        for i, (X_train, y_train) in enumerate(dataset_iter):
            model = self.model_fn()
            model.fit(X_train, y_train, **(fit_kwargs or {}))
            self._save_model(i, model)

    def _split_data_by_class(self, X_total, y, classes):
        """
        Splits the data by class for training individual attack models.

        Parameters:
        - X_total, y: Data and labels from shadow models.
        - classes: Class data from shadow models.
        """
        data_indices = np.arange(X_total.shape[0])
        return [
            (X_total[self._get_class_indices(classes, i, data_indices)], y[self._get_class_indices(classes, i, data_indices)])
            for i in range(self.num_classes)
        ]

    def _get_class_indices(self, classes, class_index, data_indices):
        """
        Gets indices of data belonging to a specific class.

        Parameters:
        - classes: Class data from shadow models.
        - class_index: Index of the class.
        - data_indices: Indices of the dataset.
        """
        if self.class_one_hot_coded:
            return data_indices[np.argmax(classes, axis=1) == class_index]
        else:
            return data_indices[np.squeeze(classes) == class_index]

    def _save_model(self, model_index, model):
        """
        Saves the model using the serializer, if available.

        Parameters:
        - model_index: Index of the model.
        - model: The model to be saved.
        """
        if self.serializer is not None:
            model_id = AttackModelBundle.MODEL_ID_FMT % model_index
            self.serializer.save(model_id, model)
        else:
            self.attack_models_.append(model)

    def predict_proba(self, X):
        """
        Predicts probabilities using the trained attack models.

        Parameters:
        - X: Input data for prediction.

        Returns:
        Array of predicted probabilities.
        """
        result = np.zeros((X.shape[0], 2))
        shadow_preds, classes = X[:, :self.num_classes], X[:, self.num_classes:]
        data_indices = np.arange(shadow_preds.shape[0])

        for i in range(self.num_classes):
            model = self._get_model(i)
            class_indices = self._get_class_indices(classes, i, data_indices)
            membership_preds = model.predict(shadow_preds[class_indices])

            for j, example_index in enumerate(class_indices):
                prob = np.squeeze(membership_preds[j])
                result[example_index] = [1 - prob, prob]

        return result

    def predict(self, X):
        """
        Predicts class membership using the trained attack models.

        Parameters:
        - X: Input data for prediction.

        Returns:
        Array of predicted class membership.
        """
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def _get_model(self, model_index):
        """
        Loads a model by its index.

        Parameters:
        - model_index: Index of the model.

        Returns:
        The loaded model.
        """
        if self.serializer is not None:
            model_id = AttackModelBundle.MODEL_ID_FMT % model_index
            return self.serializer.load(model_id)
        else:
            return self.attack_models_[model_index]
