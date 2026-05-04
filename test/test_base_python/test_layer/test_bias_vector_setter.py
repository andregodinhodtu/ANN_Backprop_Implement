import sys
sys.path.append("src/base_python_version")

import pytest
import numpy as np
from ANN_layer_base_python import ANN_Layer_base_python as ANN_Layer


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
# biases setter — happy path
# ============================================================

def test_biases_setter_list():
    
    # simple test to setter
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.biases = [[1], [0]]
    assert_equal(layer.biases, [[1], [0]])


def test_biases_setter_rejects_numpy():
    
    # Base Python implementation must reject Numpy arrays.
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(TypeError):
        layer.biases = np.array([[1], [0]])


# ============================================================
# biases setter — TypeError
# ============================================================

@pytest.mark.parametrize("bad_biases", [
    "not a list",           # string
    123,                    # int
    [1, 2],                 # flat list, not a nested list
    [[1], ["a"]],           # non-numeric values
])
def test_biases_setter_wrong_type(bad_biases):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(TypeError):
        layer.biases = bad_biases


# ============================================================
# biases setter — ValueError
# ============================================================

@pytest.mark.parametrize("bad_biases", [
    [[1], [0], [1]],        # too many rows
    [[1, 0], [0, 1]],       # more than 1 column "batch" not accepted
    [[1]],                  # too few rows
])
def test_biases_setter_wrong_shape(bad_biases):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(ValueError):
        layer.biases = bad_biases
