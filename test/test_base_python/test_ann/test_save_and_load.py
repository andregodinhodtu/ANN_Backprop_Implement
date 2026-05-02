import sys
sys.path.append("src/base_python_version")
import random
import pytest
from pathlib import Path
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


def assert_matrix_close(actual, expected, tol=1e-9):
    """Assert two list-of-lists matrices are equal within a tolerance."""
    assert len(actual) == len(expected)
    for row_a, row_e in zip(actual, expected):
        assert len(row_a) == len(row_e)
        for a, e in zip(row_a, row_e):
            assert a == pytest.approx(e, abs=tol)


# ============================================================
# save_model + load_model — round trip
# ============================================================
def test_save_and_load_round_trip(tmp_path):
    """Save a model, load it back, and verify it produces identical predictions."""
    ann = make_ann(seed=42)

    # Training-time attributes that save_model writes to the file.
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

    # Weights and biases match (within precision of the 10-decimal save format)
    for orig_layer, loaded_layer in zip(ann.layers, loaded.layers):
        assert_matrix_close(loaded_layer.weights, orig_layer.weights)
        assert_matrix_close(loaded_layer.biases, orig_layer.biases)

    # Functional equivalence: same input → same prediction
    rng = random.Random(0)
    x = [[rng.gauss(0, 1)] for _ in range(4)]
    y_orig   = ann.prediction(x)
    y_loaded = loaded.prediction(x)
    assert_matrix_close(y_loaded, y_orig)


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
