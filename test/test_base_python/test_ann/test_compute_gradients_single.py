import sys
sys.path.append("src/base_python_version")
import random
import copy
import pytest
from ANN_base_python import ANN_base_python as ANN


def make_ann(seed=0):
    """Build a small reproducible 4 → 5 → 2 network."""
    rng = random.Random(seed)
    return ANN(
        n_layers=3,
        n_neurons_each_layer=[4, 5, 2],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="binarycrossentropy",
        rng=rng,
    )


# ============================================================
# compute_gradients_sample — happy path
# ============================================================
def test_compute_gradients_sample_sets_dweights_and_dbiases():
    ann = make_ann()
    x = [[1.0], [1.0], [1.0], [1.0]]
    y = [[1.0], [0.0]]

    ann.compute_gradients_sample(x, y)

    for layer in ann.layers:
        assert layer.dweights is not None
        assert layer.dbiases is not None


def test_compute_gradients_sample_shapes():
    """dweights matches weights shape; dbiases matches biases shape."""
    ann = make_ann()
    x = [[1.0], [1.0], [1.0], [1.0]]
    y = [[1.0], [0.0]]

    ann.compute_gradients_sample(x, y)

    for layer in ann.layers:
        # dweights: (n_out, n_in)
        assert len(layer.dweights) == len(layer.weights)
        for dw_row, w_row in zip(layer.dweights, layer.weights):
            assert len(dw_row) == len(w_row)

        # dbiases: (n_out, 1)
        assert len(layer.dbiases) == len(layer.biases)
        for db_row, b_row in zip(layer.dbiases, layer.biases):
            assert len(db_row) == len(b_row)


# ============================================================
# compute_gradients_sample — gradient direction sanity
# ============================================================
def test_compute_gradients_sample_decreases_loss():
    """A small step in the negative-gradient direction should reduce the loss."""
    ann = make_ann(seed=42)
    x = [[0.5], [-0.3], [0.1], [0.8]]
    y = [[1.0], [0.0]]

    # compute_loss for base-Python takes lists of column vectors
    loss_before = ann.compute_loss([x], [y])

    ann.compute_gradients_sample(x, y)

    # Take a tiny manual gradient step on every layer.
    # We write to the backing fields directly to skip setter validation
    # and avoid in-place trouble.
    lr = 1e-3
    for layer in ann.layers:
        new_w = [
            [layer.weights[i][j] - lr * layer.dweights[i][j]
             for j in range(layer.n_neurons_input)]
            for i in range(layer.n_neurons_output)
        ]
        new_b = [
            [layer.biases[i][0] - lr * layer.dbiases[i][0]]
            for i in range(layer.n_neurons_output)
        ]
        layer._weights = new_w
        layer._biases  = new_b

    loss_after = ann.compute_loss([x], [y])
    assert loss_after < loss_before


# ============================================================
# compute_gradients_sample — ValueError cases
# ============================================================
def test_compute_gradients_sample_rejects_wrong_input_dim():
    ann = make_ann()
    # Network expects 4 input features, give it 3
    with pytest.raises(ValueError):
        ann.compute_gradients_sample([[1.0], [2.0], [3.0]], [[1.0], [0.0]])


def test_compute_gradients_sample_rejects_wrong_target_dim():
    ann = make_ann()
    # Network has 2 output neurons, give it 3
    with pytest.raises(ValueError):
        ann.compute_gradients_sample([[1.0], [1.0], [1.0], [1.0]], [[1.0], [0.0], [1.0]])


def test_compute_gradients_sample_rejects_empty_input():
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.compute_gradients_sample([], [[1.0], [0.0]])
