import numpy
import pandas
import typing

from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, ShuffleSplit


class MIA(BaseEstimator):
    """
    Membership Inference Attack (MIA) class using scikit-learn BaseEstimator.

    Attributes:
    - factory (typing.Callable[[], BaseEstimator]): A callable that creates an instance of the target model.
    - shadows (int): Number of shadow models.
    - categories (int): Number of categories in the target model's output.
    - attack (BaseEstimator): Trained attack model.

    Methods:
    - fit(X: ArrayLike, y: ArrayLike) -> BaseEstimator: Fit the attack model using the specified dataset.
    - predict(X: ArrayLike) -> ArrayLike: Make predictions using the trained attack model.
    - score(X: ArrayLike, y: ArrayLike) -> ArrayLike: Score the attack model on the provided dataset.

    Constants:
    - SPLIT_SIZE (float): Default split size for shadow models, set to 0.25.
    """

    SPLIT_SIZE = 0.25

    def __init__(
        self,
        factory: typing.Callable[[], BaseEstimator],
        categories: int,
        shadows: int = 20,
    ):
        """
        Initialize the MIA instance.

        Parameters:
        - factory (typing.Callable[[], BaseEstimator]): A callable that creates an instance of the target model.
        - categories (int): Number of categories in the target model's output.
        - shadows (int, optional): Number of shadow models to train. Defaults to 20.
        """
        self.factory = factory
        self.shadows = shadows
        self.categories = categories

    def fit(self, X: ArrayLike, y: ArrayLike) -> BaseEstimator:
        """
        Fit the MIA attack model using the specified dataset.

        Parameters:
        - X (ArrayLike): Input features.
        - y (ArrayLike): Target labels.

        Returns:
        - self (BaseEstimator): Fitted MIA instance.
        """
        iterator = ShuffleSplit(
            n_splits=self.shadows,
            test_size=self.SPLIT_SIZE,
            train_size=self.SPLIT_SIZE,
        )

        X_shadow = numpy.empty((0, self.categories))
        y_shadow = numpy.empty((0,))

        for inside_index, outside_index in iterator.split(X):
            X_train, y_train = X[inside_index], y[inside_index]
            model = self.factory().fit(X_train, y_train)

            inside_predictions = model.predict_proba(X[inside_index])
            outside_predictions = model.predict_proba(X[outside_index])

            inside_labels = numpy.zeros(inside_predictions.shape[0])
            outside_labels = numpy.ones(outside_predictions.shape[0])

            X_shadow = numpy.vstack([X_shadow, inside_predictions, outside_predictions])
            y_shadow = numpy.hstack([y_shadow, inside_labels, outside_labels])

        self.attack = self.factory().fit(X_shadow, y_shadow)

        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Make predictions using the trained attack model.

        Parameters:
        - X (ArrayLike): Input features.

        Returns:
        - predictions (ArrayLike): Model predictions.
        """
        return self.attack(X)

    def score(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """
        Score the attack model on the provided dataset.

        Parameters:
        - X (ArrayLike): Input features.
        - y (ArrayLike): Target labels.

        Returns:
        - scores (ArrayLike): Model scores.
        """
        return self.attack.score(X, y)
