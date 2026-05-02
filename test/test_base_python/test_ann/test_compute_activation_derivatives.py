import sys
sys.path.append("src/base_python_version")

import math
import pytest
from ANN_layer_base_python import ANN_Layer_base_python as ANN_Layer


# ============================================================
# Helper
# ============================================================

def assert_close(actual, expected):
    """Assert two flat lists are element-wise close."""
    assert len(actual) == len(expected), (
        f"Length mismatch: got {len(actual)}, expected {len(expected)}"
    )
    for a, e in zip(actual, expected):
        assert a == pytest.approx(e)


# ============================================================
# compute_activation_derivatives — happy path
# ============================================================
# NOTE: In the base-Python version, activation_derivatives is a flat list
# of floats (one per neuron), not a column vector.

def test_compute_derivatives_relu():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]
    layer.forward([[3], [5]])
    derivs = layer.compute_activation_derivatives()
    # z_s = [[3], [5]], both positive so derivatives should be 1
    assert_close(derivs, [1, 1])


def test_compute_derivatives_relu_negative():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[-5], [-5]]
    layer.forward([[1], [1]])
    derivs = layer.compute_activation_derivatives()
    # z_s = [[-4], [-4]], both negative so derivatives should be 0
    assert_close(derivs, [0, 0])


def test_compute_derivatives_sigmoid():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="sigmoid")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]
    layer.forward([[1], [2]])
    derivs = layer.compute_activation_derivatives()

    def sigmoid_deriv(z):
        s = 1 / (1 + math.exp(-z))
        return s * (1 - s)

    expected = [sigmoid_deriv(1), sigmoid_deriv(2)]
    assert_close(derivs, expected)


def test_compute_derivatives_leaky_relu():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="leaky_relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]
    layer.forward([[3], [-2]])
    derivs = layer.compute_activation_derivatives()
    # z_s = [[3], [-2]], positive -> 1, negative -> 0.01
    assert_close(derivs, [1, 0.01])


# ============================================================
# compute_activation_derivatives — output shape
# ============================================================

def test_compute_derivatives_shape():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=3, activation_function="relu")
    layer.weights = [[1, 0], [0, 1], [1, 1]]
    layer.biases  = [[0], [0], [0]]
    layer.forward([[1], [2]])
    derivs = layer.compute_activation_derivatives()
    # Flat list of length n_neurons_output
    assert len(derivs) == 3


# ============================================================
# compute_activation_derivatives — stored after call
# ============================================================

def test_compute_derivatives_stored():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]
    layer.forward([[1], [2]])
    layer.compute_activation_derivatives()
    assert layer.activation_derivatives is not None


# ============================================================
# compute_activation_derivatives — z_s is column vector
# ============================================================

def test_compute_derivatives_z_s_column_vector_shape():
    """z_s is stored as a column vector (list of single-element lists)."""
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=3, activation_function="relu")
    layer.weights = [[1, 0], [0, 1], [1, 1]]
    layer.biases  = [[0], [0], [0]]
    layer.forward([[1], [2]])

    assert len(layer.z_s) == 3
    assert all(len(row) == 1 for row in layer.z_s)

    derivs = layer.compute_activation_derivatives()
    assert len(derivs) == 3


# ============================================================
# compute_activation_derivatives — state tests
# ============================================================

def test_compute_derivatives_without_forward():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]
    with pytest.raises(ValueError):
        layer.compute_activation_derivatives()
