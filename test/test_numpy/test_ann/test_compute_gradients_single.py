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
# compute_gradients_sample — happy path
# ============================================================
def test_compute_gradients_sample_sets_dweights_and_dbiases():
    ann = make_ann()
    x = np.ones((4, 1))
    y = np.array([[1.0], [0.0]])

    ann.compute_gradients_sample(x, y)

    for layer in ann.layers:
        assert layer.dweights is not None
        assert layer.dbiases is not None


def test_compute_gradients_sample_shapes():
    """dweights matches weights shape; dbiases matches biases shape."""
    ann = make_ann()
    x = np.ones((4, 1))
    y = np.array([[1.0], [0.0]])

    ann.compute_gradients_sample(x, y)

    for layer in ann.layers:
        assert layer.dweights.shape == layer.weights.shape
        assert layer.dbiases.shape == layer.biases.shape


# ============================================================
# compute_gradients_sample — gradient direction sanity
# ============================================================
def test_compute_gradients_sample_decreases_loss():
    """A small step in the negative-gradient direction should reduce the loss."""
    ann = make_ann(seed=42)
    x = np.array([[0.5], [-0.3], [0.1], [0.8]])
    y = np.array([[1.0], [0.0]])

    loss_before = ann.compute_loss(x, y)

    ann.compute_gradients_sample(x, y)

    # Take a tiny manual gradient step on every layer
    lr = 1e-3
    for layer in ann.layers:
        layer._weights = layer._weights - lr * layer.dweights
        layer._biases  = layer._biases  - lr * layer.dbiases

    loss_after = ann.compute_loss(x, y)
    assert loss_after < loss_before


# ============================================================
# compute_gradients_sample — TypeError cases
# ============================================================
def test_compute_gradients_sample_rejects_non_ndarray_input():
    ann = make_ann()
    with pytest.raises(TypeError):
        ann.compute_gradients_sample([[1.0]] * 4, np.zeros((2, 1)))


def test_compute_gradients_sample_rejects_non_ndarray_target():
    ann = make_ann()
    with pytest.raises(TypeError):
        ann.compute_gradients_sample(np.zeros((4, 1)), [[1.0], [0.0]])


def test_compute_gradients_sample_rejects_non_numeric_input():
    ann = make_ann()
    bad = np.array([["a"]] * 4)
    with pytest.raises(TypeError):
        ann.compute_gradients_sample(bad, np.zeros((2, 1)))


def test_compute_gradients_sample_rejects_non_numeric_target():
    ann = make_ann()
    bad = np.array([["a"], ["b"]])
    with pytest.raises(TypeError):
        ann.compute_gradients_sample(np.zeros((4, 1)), bad)


# ============================================================
# compute_gradients_sample — ValueError cases
# ============================================================
def test_compute_gradients_sample_rejects_batch_input():
    """Function is single-sample only; batch should raise."""
    ann = make_ann()
    x = np.zeros((4, 3))   # batch of 3
    y = np.zeros((2, 1))
    with pytest.raises(ValueError):
        ann.compute_gradients_sample(x, y)


def test_compute_gradients_sample_rejects_batch_target():
    ann = make_ann()
    x = np.zeros((4, 1))
    y = np.zeros((2, 3))
    with pytest.raises(ValueError):
        ann.compute_gradients_sample(x, y)


def test_compute_gradients_sample_rejects_1d_input():
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.compute_gradients_sample(np.zeros(4), np.zeros((2, 1)))


def test_compute_gradients_sample_rejects_wrong_input_dim():
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.compute_gradients_sample(np.zeros((3, 1)), np.zeros((2, 1)))


def test_compute_gradients_sample_rejects_wrong_target_dim():
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.compute_gradients_sample(np.zeros((4, 1)), np.zeros((3, 1)))