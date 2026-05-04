import sys
sys.path.append("src/base_python_version")
import random
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
# _compute_deltas — happy path
# ============================================================
def test_compute_deltas_sets_delta_on_every_layer():
    ann = make_ann()
    x = [[1.0], [1.0], [1.0], [1.0]]
    y = [[1.0], [0.0]]
    
    # populates a_s and z_s on each layer
    ann.prediction(x)
    ann._compute_deltas(y)

    for layer in ann.layers:
        assert layer.delta is not None


def test_compute_deltas_lengths_match_layer_outputs():
    ann = make_ann()
    x = [[1.0], [1.0], [1.0], [1.0]]
    y = [[1.0], [0.0]]

    ann.prediction(x)
    ann._compute_deltas(y)
    
    # deltas are expected to be stored in a flat list
    for layer in ann.layers:
        assert len(layer.delta) == layer.n_neurons_output


def test_compute_deltas_output_layer_bce_sigmoid_shortcut():
    ann = make_ann(seed=42)
    x = [[0.5], [-0.3], [0.1], [0.8]]
    y = [[1.0], [0.0]]

    a_pred = ann.prediction(x)
    ann._compute_deltas(y)
    
    "BCE + sigmoid output, output-layer delta should equal (a - y)"
    output_layer = ann.layers[-1]
    expected = [a_pred[j][0] - y[j][0] for j in range(output_layer.n_neurons_output)]

    for actual, exp in zip(output_layer.delta, expected):
        assert actual == pytest.approx(exp)


# ============================================================
# _compute_deltas — ValueError cases
# ============================================================
def test_compute_deltas_rejects_wrong_output_dim():
    ann = make_ann()
    ann.prediction([[1.0], [1.0], [1.0], [1.0]])

    # Network has 2 output neurons; pass 3
    with pytest.raises(ValueError):
        ann._compute_deltas([[0.0], [0.0], [0.0]])
