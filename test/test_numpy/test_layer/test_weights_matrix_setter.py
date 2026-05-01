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
# weights_matrix setter — happy path
# ============================================================

def test_weights_matrix_setter_list():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    assert_equal(layer.weights, [[1, 0], [0, 1]])


def test_weights_matrix_setter_numpy():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = np.array([[1, 0], [0, 1]])
    assert_equal(layer.weights, [[1, 0], [0, 1]])


# ============================================================
# weights_matrix setter — TypeError
# ============================================================

@pytest.mark.parametrize("bad_weights", [
    "not a matrix",         # string
    123,                    # int
    [1, 2, 3],              # flat list, not list of lists
])
def test_weights_matrix_setter_wrong_type(bad_weights):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(TypeError):
        layer.weights_matrix = bad_weights


# ============================================================
# weights_matrix setter — ValueError
# ============================================================

@pytest.mark.parametrize("bad_weights", [
    [[1, 0], [0, 1], [1, 1]],   # too many rows
    [[1, 0, 1], [0, 1, 0]],     # too many columns
    [[1], [0]],                  # too few columns
])
def test_weights_matrix_setter_wrong_shape(bad_weights):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(ValueError):
        layer.weights_matrix = bad_weights