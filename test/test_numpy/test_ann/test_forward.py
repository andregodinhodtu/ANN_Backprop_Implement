import sys
sys.path.append("src/numpy_version")
import pytest
import numpy as np
from ANN_numpy import ANN_numpy as ANN


# ============================================================
# Helper: build a small standard network for reuse
# ============================================================
def make_ann(seed = 0):
    """Build a small reproducible 4 → 5 → 2 network."""
    rng = np.random.default_rng(seed)
    return ANN(
        n_layers=3,
        n_neurons_each_layer=[4, 5, 2],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="binarycrossentropy",
        rng = rng,
    )

# ============================================================
# prediction — happy path: single sample
# ============================================================
def test_prediction_single_sample_shape():
    ann = make_ann()
    x = np.zeros((4, 1))
    y = ann.prediction(x)

    assert isinstance(y, np.ndarray)
    assert y.shape == (2, 1)


def test_prediction_single_sample_sigmoid_range():
    
    # Output activation is sigmoid, so all outputs must be in (0, 1).
    ann = make_ann()
    x = np.random.default_rng(0).normal(size=(4, 1))
    y = ann.prediction(x)

    assert np.all(y > 0)
    assert np.all(y < 1)


# ============================================================
# prediction — happy path: batch
# ============================================================
def test_prediction_batch_shape():
    ann = make_ann()
    batch_size = 8
    x = np.zeros((4, batch_size))
    y = ann.prediction(x)

    assert y.shape == (2, batch_size)


# ============================================================
# prediction — TypeError cases
# ============================================================
def test_prediction_rejects_non_ndarray():
    ann = make_ann()
    with pytest.raises(TypeError):
        ann.prediction([[1.0], [2.0], [3.0], [4.0]])  # list, not ndarray


def test_prediction_rejects_non_numeric_dtype():
    ann = make_ann()
    bad = np.array([["a"], ["b"], ["c"], ["d"]])  # dtype is '<U1', not numeric
    with pytest.raises(TypeError):
        ann.prediction(bad)


# ============================================================
# prediction — ValueError cases (shape / dimension)
# ============================================================
def test_prediction_rejects_1d_array():
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.prediction(np.zeros(4))  # 1D, not 2D


def test_prediction_rejects_3d_array():
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.prediction(np.zeros((4, 1, 1)))  # 3D


def test_prediction_rejects_zero_batch():
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.prediction(np.zeros((4, 0)))  # batch_size = 0


def test_prediction_rejects_wrong_feature_dim():
    ann = make_ann()
    # Network expects 4 features, give it 3
    with pytest.raises(ValueError):
        ann.prediction(np.zeros((3, 1)))


def test_prediction_rejects_transposed_batch():
    
    # A common user mistake: passing (batch_size, n_features) instead of (n_features, batch_size).
    ann = make_ann()
    # batch of 7 samples, 4 features each, but transposed → shape (7, 4)
    bad = np.zeros((7, 4))
    with pytest.raises(ValueError):
        ann.prediction(bad)


# ============================================================
# prediction — determinism with a seed
# ============================================================
def test_prediction_is_deterministic_with_same_seed():
    ann1 = make_ann(seed=123)
    ann2 = make_ann(seed=123)
    x = np.ones((4, 3))

    np.testing.assert_allclose(ann1.prediction(x), ann2.prediction(x))