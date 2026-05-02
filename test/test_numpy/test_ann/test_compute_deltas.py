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
# _compute_deltas — happy path
# ============================================================
def test_compute_deltas_sets_delta_on_every_layer():
    ann = make_ann()
    x = np.ones((4, 1))
    y = np.array([[1.0], [0.0]])

    ann.prediction(x)            # populates a_s and z_s on each layer
    ann._compute_deltas(y)

    for layer in ann.layers:
        assert layer.delta is not None


def test_compute_deltas_shapes_match_layer_outputs():
    ann = make_ann()
    batch_size = 3
    x = np.ones((4, batch_size))
    y = np.zeros((2, batch_size))

    ann.prediction(x)
    ann._compute_deltas(y)

    for layer in ann.layers:
        assert layer.delta.shape == (layer.n_neurons_output, batch_size)


def test_compute_deltas_batch_matches_single():
    """A delta computed for a batch should equal stacking single-sample deltas."""
    ann = make_ann(seed=42)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 4))
    y = rng.integers(0, 2, size=(2, 4)).astype(float)

    # Batched
    ann.prediction(x)
    ann._compute_deltas(y)
    batched_deltas = [layer.delta.copy() for layer in ann.layers]

    # Single samples, one column at a time
    for i in range(x.shape[1]):
        ann.prediction(x[:, i:i+1])
        ann._compute_deltas(y[:, i:i+1])
        for layer_idx, layer in enumerate(ann.layers):
            np.testing.assert_allclose(
                layer.delta,
                batched_deltas[layer_idx][:, i:i+1],
                rtol=1e-10,
            )


# ============================================================
# _compute_deltas — ValueError cases
# ============================================================
def test_compute_deltas_rejects_1d_y():
    ann = make_ann()
    ann.prediction(np.ones((4, 1)))

    with pytest.raises(ValueError):
        ann._compute_deltas(np.array([1.0, 0.0]))   # 1D


def test_compute_deltas_rejects_wrong_output_dim():
    ann = make_ann()
    ann.prediction(np.ones((4, 1)))

    # Network has 2 output neurons; pass 3
    with pytest.raises(ValueError):
        ann._compute_deltas(np.zeros((3, 1)))