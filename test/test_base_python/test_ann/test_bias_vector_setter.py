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
# biases setter — happy path
# ============================================================

def test_biases_setter_list():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.biases = [[1], [0]]
    assert_equal(layer.biases, [[1], [0]])


def test_biases_setter_floats():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.biases = [[0.5], [-1.2]]
    assert_equal(layer.biases, [[0.5], [-1.2]])


# ============================================================
# biases setter — TypeError
# ============================================================

@pytest.mark.parametrize("bad_biases", [
    "not a list",       # string
    123,                # int
    [1, 2],             # flat list, not list of lists
    [[1], ["a"]],       # non-numeric values
])
def test_biases_setter_wrong_type(bad_biases):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(TypeError):
        layer.biases = bad_biases


# ============================================================
# biases setter — ValueError
# ============================================================

@pytest.mark.parametrize("bad_biases", [
    [[1], [0], [1]],     # too many rows
    [[1, 0], [0, 1]],    # more than 1 column
    [[1]],               # too few rows
])
def test_biases_setter_wrong_shape(bad_biases):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(ValueError):
        layer.biases = bad_biases
