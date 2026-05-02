import sys
sys.path.append("src/base_python_version")

import pytest
import numpy as np
from ANN_layer_base_python import ANN_Layer_base_python as ANN_Layer


# ============================================================
# Helper: assert both shape and values
# ============================================================

def assert_matrix_equal(actual, expected):
    """Assert that actual and expected have the same shape and close values."""
    actual_arr = np.array(actual)
    expected_arr = np.array(expected)
    assert actual_arr.shape == expected_arr.shape, (
        f"Shape mismatch: got {actual_arr.shape}, expected {expected_arr.shape}"
    )
    assert np.allclose(actual_arr, expected_arr)


# ============================================================
# weights setter — happy path
# ============================================================

def test_weights_setter_list():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    assert_matrix_equal(layer.weights, [[1, 0], [0, 1]])


def test_weights_setter_rejects_numpy():
    """Base Python implementation must reject numpy arrays."""
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(TypeError):
        layer.weights = np.array([[1, 0], [0, 1]])


# ============================================================
# weights setter — TypeError
# ============================================================

@pytest.mark.parametrize("bad_weights", [
    "not a matrix",         # string
    123,                    # int
    [1, 2, 3],              # flat list, not list of lists
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
    [[1], [0]],                  # too few columns
])
def test_weights_setter_wrong_shape(bad_weights):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(ValueError):
        layer.weights = bad_weights
