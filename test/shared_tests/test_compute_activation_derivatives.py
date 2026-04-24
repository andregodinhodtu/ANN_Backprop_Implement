import pytest
import sys
import numpy as np
sys.path.append("src/base_python_version")
sys.path.append("src/numpy_version")
from ANN_layer_base_python import ANN_Layer_base_python
from ANN_layer_numpy import ANN_Layer_numpy

# ============================================================
# Fixture
# ============================================================

@pytest.fixture(params=["base_python", "numpy"])
def ANN_Layer(request):
    if request.param == "base_python":
        return ANN_Layer_base_python
    return ANN_Layer_numpy

# ============================================================
# compute_activation_derivatives — happy path
# ============================================================

def test_compute_derivatives_relu(ANN_Layer):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]
    layer.forward([[3], [5]])
    derivs = layer.compute_activation_derivatives()
    # z_s = [[3], [5]], both positive so derivatives should be 1
    assert np.allclose(derivs, [[1], [1]])

def test_compute_derivatives_relu_negative(ANN_Layer):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[-5], [-5]]
    layer.forward([[1], [1]])
    derivs = layer.compute_activation_derivatives()
    # z_s = [[-4], [-4]], both negative so derivatives should be 0
    assert np.allclose(derivs, [[0], [0]])

def test_compute_derivatives_sigmoid(ANN_Layer):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="sigmoid")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]
    layer.forward([[1], [2]])
    derivs = layer.compute_activation_derivatives()
    # sigmoid derivative: s(z) * (1 - s(z))
    def sigmoid_deriv(z):
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)
    expected = [[sigmoid_deriv(1)], [sigmoid_deriv(2)]]
    assert np.allclose(derivs, expected)

def test_compute_derivatives_leaky_relu(ANN_Layer):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="leaky_relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]
    layer.forward([[3], [-2]])
    derivs = layer.compute_activation_derivatives()
    # z_s = [[3], [-2]], positive -> 1, negative -> 0.01
    assert np.allclose(derivs, [[1], [0.01]])

# ============================================================
# compute_activation_derivatives — output shape
# ============================================================

def test_compute_derivatives_shape(ANN_Layer):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=3, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1], [1, 1]]
    layer.biases_vector  = [[0], [0], [0]]
    layer.forward([[1], [2]])
    derivs = layer.compute_activation_derivatives()
    assert np.array(derivs).shape == (3, 1)

# ============================================================
# compute_activation_derivatives — stored after call
# ============================================================

def test_compute_derivatives_stored(ANN_Layer):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]
    layer.forward([[1], [2]])
    layer.compute_activation_derivatives()
    assert layer.activation_derivatives is not None

# ============================================================
# compute_activation_derivatives — state tests
# ============================================================

def test_compute_derivatives_without_forward(ANN_Layer):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]
    with pytest.raises(ValueError):
        layer.compute_activation_derivatives()