import numpy

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from typing import Callable


class Shadow(BaseEstimator):

    def __init__(self, factory: Callable, shadows: int = 20):
        self.factory = factory
        self.shadows = shadows

    def fit(self, X: ArrayLike, y: ArrayLike) -> Shadow:
        self.models = []
        self.inside = []
        self.outside = []

        iterator = ShuffleSplit(n_splits=self.shadows, test_size=0.5)

        for inside, outside in iterator.split(X):
            X_train, y_train = X[inside], y[inside]

            model = self.factory()
            model.fit(X_train, y_train)
            
            self.models.append(model)
            self.inside.append(inside)
            self.outside.append(outside)

        self.X_fit_ = X
        self.y_fit_ = y

        return self

    def transform(self) -> tuple[ArrayLike, ArrayLike]:
        self.query = []
        self.labels = []
        self.original = []

        iterator = zip(self.model, self.inside, self.outside)

        for model, inside, outside in iterator:
            inside_predictions = model.predict_proba(inside)
            outside_predictions = mode.predict_proba(outside)

            inside_labels = numpy.zeros(inside_predictions.shape[0])
            outside_labels = numpy.ones(outside_labels.shape[0])

            inside_original = self.y_fit_[inside]
            outside_original = self.y_fit_[outside]

            query = numpy.vstack([inside_predictions, outside_predictions])
            labels = np.hstack([inside_labels, outside_labels])
            original = np.hstack([inside_original, outside_original])

            self.query.append(query)
            self.labels.append(labels)
            self.original.append(original)

        X_shadow = numpy.hstack(
            np.hstack(self.original).reshape(-1, 1),
            np.vstack(self.query)
        )

        y_shadow = numpy.hstack(self.original)

        return X_shadow, y_shadow
