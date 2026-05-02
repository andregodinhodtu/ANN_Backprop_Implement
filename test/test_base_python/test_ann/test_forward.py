import sys
sys.path.append("src/base_python_version")
import random
import pytest
from ANN_base_python import ANN_base_python as ANN


# ============================================================
# Helper: build a small standard network for reuse
# ============================================================
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
# prediction — happy path
# ============================================================
def test_prediction_single_sample_shape():
    ann = make_ann()
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = ann.prediction(x)

    # Output is a list-of-lists, shape (2, 1)
    assert isinstance(y, list)
    assert len(y) == 2
    assert all(len(row) == 1 for row in y)


def test_prediction_single_sample_sigmoid_range():
    """Output activation is sigmoid, so all outputs must be in (0, 1)."""
    ann = make_ann()
    rng = random.Random(0)
    x = [[rng.gauss(0, 1)] for _ in range(4)]
    y = ann.prediction(x)

    for row in y:
        for val in row:
            assert 0 < val < 1


# ============================================================
# prediction — TypeError cases
# ============================================================
def test_prediction_rejects_non_list():
    ann = make_ann()
    with pytest.raises(TypeError):
        ann.prediction("not a list")


def test_prediction_rejects_rows_not_lists():
    ann = make_ann()
    with pytest.raises(TypeError):
        ann.prediction([1.0, 2.0, 3.0, 4.0])  # flat list, rows aren't lists


def test_prediction_rejects_non_numeric_values():
    ann = make_ann()
    with pytest.raises(TypeError):
        ann.prediction([["a"], ["b"], ["c"], ["d"]])


# ============================================================
# prediction — ValueError cases (shape / dimension)
# ============================================================
def test_prediction_rejects_empty_input():
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.prediction([])


def test_prediction_rejects_multi_column_rows():
    """Each row in input_vector must contain exactly 1 element."""
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.prediction([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


def test_prediction_rejects_wrong_feature_dim():
    """Network expects 4 features; pass 3."""
    ann = make_ann()
    with pytest.raises(ValueError):
        ann.prediction([[1.0], [2.0], [3.0]])


# ============================================================
# prediction — determinism with a seed
# ============================================================
def test_prediction_is_deterministic_with_same_seed():
    ann1 = make_ann(seed=123)
    ann2 = make_ann(seed=123)
    x = [[1.0], [1.0], [1.0], [1.0]]

    y1 = ann1.prediction(x)
    y2 = ann2.prediction(x)

    for row1, row2 in zip(y1, y2):
        for v1, v2 in zip(row1, row2):
            assert v1 == pytest.approx(v2)
