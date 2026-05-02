import sys
sys.path.append("src/numpy_version")
import pytest
import numpy as np
from ANN_numpy import ANN_numpy as ANN


def make_ann(seed = 0):
    
    rng = np.random.default_rng(seed)
    """Build a small reproducible 4 → 5 → 2 network."""
    return ANN(
        n_layers=3,
        n_neurons_each_layer=[4, 5, 2],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="binarycrossentropy",
        rng = rng,
    )

# ============================================================
# compute_loss — happy path
# ============================================================
def test_compute_loss_returns_float():
    ann = make_ann()
    X = np.ones((4, 3))
    Y = np.zeros((2, 3))

    loss = ann.compute_loss(X, Y)
    assert isinstance(loss, float)


def test_compute_loss_is_non_negative():
    """BCE is always >= 0."""
    ann = make_ann()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 5))
    Y = rng.integers(0, 2, size=(2, 5)).astype(float)

    assert ann.compute_loss(X, Y) >= 0.0


def test_compute_loss_single_vs_batch_consistency():
    """Mean loss over a batch == mean of per-sample losses."""
    ann = make_ann(seed=42)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 6))
    Y = rng.integers(0, 2, size=(2, 6)).astype(float)

    batched = ann.compute_loss(X, Y)
    per_sample = [
        ann.compute_loss(X[:, i:i+1], Y[:, i:i+1])
        for i in range(X.shape[1])
    ]
    np.testing.assert_allclose(batched, np.mean(per_sample), rtol=1e-10)


# ============================================================
# compute_loss — TypeError cases
# ============================================================
def test_compute_loss_rejects_non_ndarray_X():
    ann = make_ann()
    with pytest.raises(TypeError):
        ann.compute_loss([[1.0]] * 4, np.zeros((2, 1)))


def test_compute_loss_rejects_non_ndarray_Y():
    ann = make_ann()
    with pytest.raises(TypeError):
        ann.compute_loss(np.zeros((4, 1)), [[0.0], [1.0]])


def test_compute_loss_rejects_non_numeric_X():
    ann = make_ann()
    bad_X = np.array([["a"]] * 4)  # dtype '<U1'
    with pytest.raises(TypeError):
        ann.compute_loss(bad_X, np.zeros((2, 1)))


def test_compute_loss_rejects_non_numeric_Y():
    ann = make_ann()
    bad_Y = np.array([["a"], ["b"]])  # dtype '<U1'
    with pytest.raises(TypeError):
        ann.compute_loss(np.zeros((4, 1)), bad_Y)


# ============================================================
# compute_loss — ValueError cases
# ============================================================
def test_compute_loss_rejects_1d_X():
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.compute_loss(np.zeros(4), np.zeros((2, 1)))


def test_compute_loss_rejects_1d_Y():
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.compute_loss(np.zeros((4, 1)), np.zeros(2))


def test_compute_loss_rejects_mismatched_batch_size():
    ann = make_ann()
    X = np.zeros((4, 5))
    Y = np.zeros((2, 3))   # different number of samples
    with pytest.raises(ValueError):
        ann.compute_loss(X, Y)


def test_compute_loss_rejects_wrong_input_features():
    ann = make_ann()
    X = np.zeros((3, 1))   # network expects 4 features
    Y = np.zeros((2, 1))
    with pytest.raises(ValueError):
        ann.compute_loss(X, Y)


def test_compute_loss_rejects_wrong_output_dim():
    ann = make_ann()
    X = np.zeros((4, 1))
    Y = np.zeros((3, 1))   # network outputs 2 neurons
    with pytest.raises(ValueError):
        ann.compute_loss(X, Y)