import numpy as np
import pytest

from mia.models import ShadowModelBundle


class DummyModel:
    def fit(self, X, y, **kwargs):
        pass

    def predict_proba(self, X):
        # Dummy implementation, return random probabilities
        return np.random.rand(len(X), 2)


def test_initialization():
    bundle = ShadowModelBundle(DummyModel, 100, num_models=10, seed=123)
    assert bundle.shadow_dataset_size == 100
    assert bundle.num_models == 10
    assert bundle.seed == 123
    assert bundle.serializer is None


def test_train_and_transform():
    X = np.random.rand(200, 10)
    y = np.random.randint(0, 2, 200)

    bundle = ShadowModelBundle(DummyModel, 50, num_models=4, seed=123)
    X_transformed, y_transformed = bundle.fit_transform(X, y)

    assert (
        X_transformed.shape[0] == 4 * 50 * 2
    )  # Assuming prepare_attack_data returns 2 samples per input
    assert y_transformed.shape[0] == 4 * 50 * 2


@pytest.mark.parametrize("model_index, expected_id", [(0, "shadow_0"), (1, "shadow_1")])
def test_model_loading(model_index, expected_id):
    class DummySerializer:
        def load(self, model_id):
            assert model_id == expected_id
            return DummyModel()

    bundle = ShadowModelBundle(
        DummyModel, 100, num_models=2, serializer=DummySerializer()
    )
    bundle._load_model(model_index)


# More tests can be added to cover other methods and edge cases
