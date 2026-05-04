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
# compute_activation_derivatives — happy path
# ============================================================

def test_compute_derivatives_relu():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]
    layer.forward([[3], [5]])
    derivs = layer.compute_activation_derivatives()
    
    # [[3], [5]], both positive so derivatives should be 1
    # derivatives are stored flat (the deltas) on purpose
    assert_equal(derivs, [1, 1])


def test_compute_derivatives_relu_negative():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[-5], [-5]]
    layer.forward([[1], [1]])
    derivs = layer.compute_activation_derivatives()
    
    # [[-4], [-4]], both negative so derivatives should be 0
    # derivatives are stored flat (the deltas) on purpose
    assert_equal(derivs, [0, 0])


def test_compute_derivatives_sigmoid():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="sigmoid")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]
    layer.forward([[1], [2]])
    derivs = layer.compute_activation_derivatives()

    # quick sigmoid helper
    def sigmoid_deriv(z):
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)

    # derivatives stored flat
    expected = [sigmoid_deriv(1), sigmoid_deriv(2)]
    assert_equal(derivs, expected)


def test_compute_derivatives_leaky_relu():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="leaky_relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]
    layer.forward([[3], [-2]])
    derivs = layer.compute_activation_derivatives()
    
    # [[3], [-2]], positive -> 1, negative -> 0.01
    assert_equal(derivs, [1, 0.01])


# ============================================================
# compute_activation_derivatives — output shape
# ============================================================

def test_compute_derivatives_shape():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=3, activation_function="relu")
    layer.weights = [[1, 0], [0, 1], [1, 1]]
    layer.biases  = [[0], [0], [0]]
    layer.forward([[1], [2]])
    derivs = layer.compute_activation_derivatives()
    
    # assert once again shape shoud be flat
    assert np.array(derivs).shape == (3,)


# ============================================================
# compute_activation_derivatives — stored after call
# ============================================================

def test_compute_derivatives_stored():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]
    layer.forward([[1], [2]])
    layer.compute_activation_derivatives()
    # stored
    assert layer.activation_derivatives is not None


# ============================================================
# compute_activation_derivatives — state tests
# ============================================================

def test_compute_derivatives_without_forward():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]
    with pytest.raises(ValueError):
        layer.compute_activation_derivatives()