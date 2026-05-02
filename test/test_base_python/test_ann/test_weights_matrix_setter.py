import sys
sys.path.append("src/base_python_version")

import pytest
from ANN_layer_base_python import ANN_Layer_base_python as ANN_Layer


# ============================================================
# Helper
# ============================================================

def assert_equal(actual, expected):
    """Assert two list-of-lists are equal (shape + values)."""
    assert len(actual) == len(expected), (
        f"Row count mismatch: got {len(actual)}, expected {len(expected)}"
    )
    for row_a, row_e in zip(actual, expected):
        assert len(row_a) == len(row_e), (
            f"Column count mismatch: got {len(row_a)}, expected {len(row_e)}"
        )
        for a, e in zip(row_a, row_e):
            assert a == pytest.approx(e)


# ============================================================
# weights setter — happy path
# ============================================================

def test_weights_setter_list():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    assert_equal(layer.weights, [[1, 0], [0, 1]])


def test_weights_setter_floats():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[0.5, -0.5], [1.5, -1.5]]
    assert_equal(layer.weights, [[0.5, -0.5], [1.5, -1.5]])


# ============================================================
# weights setter — TypeError
# ============================================================

@pytest.mark.parametrize("bad_weights", [
    "not a matrix",         # string
    123,                    # int
    [1, 2, 3],              # flat list, not list of lists
    [[1, 0], ["a", "b"]],   # non-numeric values
])
def test_weights_setter_wrong_type(bad_weights):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(TypeError):
        layer.weights = bad_weights


# ============================================================
# weights setter — ValueError
# ============================================================

@pytest.mark.parametrize("bad_weights", [
    [[1, 0], [0, 1], [1, 1]],   # too many rows
    [[1, 0, 1], [0, 1, 0]],     # too many columns
    [[1], [0]],                 # too few columns
])
def test_weights_setter_wrong_shape(bad_weights):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(ValueError):
        layer.weights = bad_weights
