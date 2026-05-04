import sys
sys.path.append("src/numpy_version")
import pytest
import numpy as np
from pathlib import Path
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
# save_model + load_model — round trip
# ============================================================
def test_save_and_load_round_trip(tmp_path):
    
    # Save a model, load it back, and verify it produces identical predictions.
    ann = make_ann(seed=42)

    # Some training-time attributes save_model writes to the file.
    # If your ANN sets these in __init__, you can skip these lines.
    ann.n_samples = 100
    ann.epochs = 10
    ann.learning_rate = 0.01
    ann.batch_size = 32
    ann.lr_decay = 0.0
    ann.decay_every = 0
    ann.l2_lambda = 0.0

    # Save
    ann.save_model(
        output_filename="test_model.txt",
        data_name="dummy_data",
        path=str(tmp_path),
    )

    saved_file = tmp_path / "test_model.txt"
    assert saved_file.exists()

    # Load
    loaded = ANN.load_model(str(saved_file))

    # Architecture matches
    assert loaded.n_layers == ann.n_layers
    assert loaded.n_neurons_each_layer == ann.n_neurons_each_layer
    assert loaded.activation_hidden == ann.activation_hidden
    assert loaded.activation_output == ann.activation_output
    assert loaded.loss_function == ann.loss_function

    # Weights and biases match (within float-precision of the 10-decimal save format)
    for orig_layer, loaded_layer in zip(ann.layers, loaded.layers):
        np.testing.assert_allclose(loaded_layer.weights, orig_layer.weights, atol=1e-9)
        np.testing.assert_allclose(loaded_layer.biases, orig_layer.biases, atol=1e-9)

    # Functional equivalence: same input → same prediction
    x = np.random.default_rng(0).normal(size=(4, 3))
    np.testing.assert_allclose(loaded.prediction(x), ann.prediction(x), atol=1e-9)


# ============================================================
# save_model — input validation
# ============================================================
def test_save_model_rejects_non_string_filename(tmp_path):
    ann = make_ann()
    ann.n_samples = ann.epochs = ann.learning_rate = ann.batch_size = 0
    ann.lr_decay = ann.decay_every = ann.l2_lambda = 0

    with pytest.raises(TypeError):
        ann.save_model(output_filename=123, data_name="x", path=str(tmp_path))


def test_save_model_rejects_non_string_data_name(tmp_path):
    ann = make_ann()
    ann.n_samples = ann.epochs = ann.learning_rate = ann.batch_size = 0
    ann.lr_decay = ann.decay_every = ann.l2_lambda = 0

    with pytest.raises(TypeError):
        ann.save_model(output_filename="m.txt", data_name=123, path=str(tmp_path))
