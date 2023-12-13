import functools
import numpy
import typing

from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import KFold


class Labeler(BaseEstimator):
    """
    A class for labeling data using a base estimator.

    Parameters:
    - factory (callable): A callable that produces an instance of BaseEstimator.

    Attributes:
    - _model (BaseEstimator): The base estimator for labeling.

    Methods:
    - fit(X, y): Fit the model to the given data.
    - predict(X): Predict labels for the given data.
    - predict_proba(X): Predict class probabilities for the given data.
    - score(X, y): Compute the accuracy score of the model on the given data.

    """

    def __init__(self, factory: typing.Callable[[], BaseEstimator]):
        self._model = factory()

    def fit(self, X: ArrayLike, y: ArrayLike) -> BaseEstimator:
        """
        Fit the model to the given data.

        Parameters:
        - X (ArrayLike): Input data.
        - y (ArrayLike): Target labels.

        Returns:
        - BaseEstimator: The fitted model.

        """
        self._model.fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict labels for the given data.

        Parameters:
        - X (ArrayLike): Input data.

        Returns:
        - ArrayLike: Predicted labels.

        """
        return self._model.predict(X)

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict class probabilities for the given data.

        Parameters:
        - X (ArrayLike): Input data.

        Returns:
        - ArrayLike: Predicted class probabilities.

        """
        prediction = self._model.predict_proba(X)
        labeled = numpy.zeros(prediction.shape)
        labeled[numpy.arange(prediction.shape[0]), numpy.argmax(prediction, axis=-1)] = 1
        return labeled

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute the accuracy score of the model on the given data.

        Parameters:
        - X (ArrayLike): Input data.
        - y (ArrayLike): Target labels.

        Returns:
        - float: Accuracy score.

        """
        return self._model.score(X, y)


class Ensemble(BaseEstimator):
    """
    A class for creating an ensemble of base estimators.

    Parameters:
    - factory (callable): A callable that produces an instance of BaseEstimator.
    - n_estimators (int): Number of base estimators in the ensemble.

    Attributes:
    - _n_estimators (int): Number of base estimators in the ensemble.
    - _models (list): List of base estimator instances.

    Methods:
    - fit(X, y): Fit the ensemble to the given data.
    - predict_proba(X): Predict class probabilities for the given data.
    - predict(X): Predict labels for the given data.
    - score(X, y): Compute the accuracy score of the ensemble on the given data.

    """

    def __init__(self, factory: typing.Callable[[], BaseEstimator], n_estimators: int):
        self._n_estimators = n_estimators
        self._models = [factory() for _ in range(n_estimators)]

    def fit(self, X: ArrayLike, y: ArrayLike) -> BaseEstimator:
        """
        Fit the ensemble to the given data using cross-validation.

        Parameters:
        - X (ArrayLike): Input data.
        - y (ArrayLike): Target labels.

        Returns:
        - BaseEstimator: The fitted ensemble.

        """
        splitter = KFold(n_splits=self._n_estimators, shuffle=True)

        for model, (train_index, test_index) in zip(self._models, splitter.split(X, y)):
            model.fit(X[train_index], y[train_index])

        return self

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict class probabilities for the given data.

        Parameters:
        - X (ArrayLike): Input data.

        Returns:
        - ArrayLike: Predicted class probabilities.

        """
        vector = numpy.random.uniform(size=(self._n_estimators,))
        vector = vector / numpy.linalg.norm(vector)

        predictions = [model.predict_proba(X) for model in self._models]
        predictions = [(pred * scale) for pred, scale in zip(predictions, vector)]

        return numpy.sum(predictions, 0)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict labels for the given data.

        Parameters:
        - X (ArrayLike): Input data.

        Returns:
        - ArrayLike: Predicted labels.

        """
        return numpy.argmax(self.predict_proba(X), axis=1)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute the accuracy score of the ensemble on the given data.

        Parameters:
        - X (ArrayLike): Input data.
        - y (ArrayLike): Target labels.

        Returns:
        - float: Accuracy score.

        """
        return accuracy_score(self.predict(X), y)
