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
# Helper to set up a layer with gradients ready
# ============================================================

def make_layer_with_gradients(ANN_Layer):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1.0, 0.0], [0.0, 1.0]]
    layer.biases_vector  = [[0.0], [0.0]]
    layer.forward([[1], [2]])
    layer.compute_activation_derivatives()
    # Manually set gradients
    layer.dweights = np.array([[0.1, 0.2], [0.3, 0.4]])
    layer.dbiases  = np.array([[0.1], [0.2]])
    return layer

# ============================================================
# update_parameters — happy path
# ============================================================

def test_update_weights_correct(ANN_Layer):
    layer = make_layer_with_gradients(ANN_Layer)
    old_weights = np.array(layer.weights).copy()
    layer.update_parameters(learning_rate=0.1)
    expected = old_weights - 0.1 * np.array([[0.1, 0.2], [0.3, 0.4]])
    assert np.allclose(layer.weights, expected)

def test_update_biases_correct(ANN_Layer):
    layer = make_layer_with_gradients(ANN_Layer)
    old_biases = np.array(layer.biases).copy()
    layer.update_parameters(learning_rate=0.1)
    expected = old_biases - 0.1 * np.array([[0.1], [0.2]])
    assert np.allclose(layer.biases, expected)

def test_update_weights_with_l2(ANN_Layer):
    layer = make_layer_with_gradients(ANN_Layer)
    old_weights = np.array(layer.weights).copy()
    layer.update_parameters(learning_rate=0.1, l2_lambda=0.01)
    expected = old_weights - 0.1 * (np.array([[0.1, 0.2], [0.3, 0.4]]) + 0.01 * old_weights)
    assert np.allclose(layer.weights, expected)

def test_l2_does_not_affect_biases(ANN_Layer):
    layer = make_layer_with_gradients(ANN_Layer)
    old_biases = np.array(layer.biases).copy()
    layer.update_parameters(learning_rate=0.1, l2_lambda=0.99)
    expected = old_biases - 0.1 * np.array([[0.1], [0.2]])
    assert np.allclose(layer.biases, expected)

# ============================================================
# update_parameters — intermediate variables cleared
# ============================================================

def test_intermediates_cleared_after_update(ANN_Layer):
    layer = make_layer_with_gradients(ANN_Layer)
    layer.update_parameters(learning_rate=0.1)
    assert layer.dweights is None
    assert layer.dbiases is None
    assert layer.delta is None
    assert layer.activation_derivatives is None
    assert layer.z_s is None
    assert layer.a_s is None

# ============================================================
# update_parameters — TypeError tests
# ============================================================

@pytest.mark.parametrize("learning_rate, l2_lambda", [
    ("0.1", 0.0),   # learning_rate wrong type
    (0.1,   "0.0"), # l2_lambda wrong type
])
def test_update_wrong_type(ANN_Layer, learning_rate, l2_lambda):
    layer = make_layer_with_gradients(ANN_Layer)
    with pytest.raises(TypeError):
        layer.update_parameters(learning_rate=learning_rate, l2_lambda=l2_lambda)

# ============================================================
# update_parameters — ValueError tests
# ============================================================

@pytest.mark.parametrize("learning_rate, l2_lambda", [
    (0.0,  0.0),    # learning_rate zero
    (-0.1, 0.0),    # learning_rate negative
    (0.1,  -0.01),  # l2_lambda negative
])
def test_update_wrong_values(ANN_Layer, learning_rate, l2_lambda):
    layer = make_layer_with_gradients(ANN_Layer)
    with pytest.raises(ValueError):
        layer.update_parameters(learning_rate=learning_rate, l2_lambda=l2_lambda)

# ============================================================
# update_parameters — state tests
# ============================================================

def test_update_without_gradients(ANN_Layer):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1.0, 0.0], [0.0, 1.0]]
    layer.biases_vector  = [[0.0], [0.0]]
    with pytest.raises(ValueError):
        layer.update_parameters(learning_rate=0.1)