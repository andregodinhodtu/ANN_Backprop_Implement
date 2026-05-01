import sys
sys.path.append("src/numpy_version")

import pytest
import numpy as np
from ANN_layer_numpy import ANN_Layer_numpy as ANN_Layer


# ============================================================
# Helper: assert both shape and values
# ============================================================

def assert_equal(actual, expected):
    """Assert that actual and expected have the same shape and close values."""
    actual_arr = np.array(actual)
    expected_arr = np.array(expected)
    assert actual_arr.shape == expected_arr.shape, (
        f"Shape mismatch: got {actual_arr.shape}, expected {expected_arr.shape}"
    )
    assert np.allclose(actual_arr, expected_arr)


# ============================================================
# biases_vector setter — happy path
# ============================================================

def test_biases_vector_setter_list():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.biases_vector = [[1], [0]]
    assert_equal(layer.biases, [[1], [0]])


def test_biases_vector_setter_numpy():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.biases_vector = np.array([[1], [0]])
    assert_equal(layer.biases, [[1], [0]])


# ============================================================
# biases_vector setter — TypeError
# ============================================================

@pytest.mark.parametrize("bad_biases", [
    "not a list",            # string
    123,                     # int
    [1, 2],                  # flat list, not list of lists
    [[1], ["a"]],            # non-numeric values
    np.array([["a"], ["b"]]) # numpy array with non-numeric values
])
def test_biases_vector_setter_wrong_type(bad_biases):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(TypeError):
        layer.biases_vector = bad_biases


# ============================================================
# biases_vector setter — ValueError
# ============================================================

@pytest.mark.parametrize("bad_biases", [
    [[1], [0], [1]],        # too many rows
    [[1, 0], [0, 1]],       # more than 1 column
    [[1]],                  # too few rows
])
def test_biases_vector_setter_wrong_shape(bad_biases):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(ValueError):
        layer.biases_vector = bad_biases
