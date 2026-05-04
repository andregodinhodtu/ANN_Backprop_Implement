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
# Helper to set up a layer with gradients ready
# ============================================================

def make_layer_with_gradients():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1.0, 0.0], [0.0, 1.0]]
    layer.biases  = [[0.0], [0.0]]
    layer.forward([[1], [2]])
    layer.compute_activation_derivatives()
    # Manually set gradients
    layer.dweights = np.array([[0.1, 0.2], [0.3, 0.4]])
    layer.dbiases  = np.array([[0.1], [0.2]])
    return layer


# ============================================================
# update_parameters — happy path
# ============================================================

def test_update_weights_correct():
    
    # testing update 
    layer = make_layer_with_gradients()
    old_weights = np.array(layer.weights).copy()
    layer.update_parameters(learning_rate=0.1)
    expected = old_weights - 0.1 * np.array([[0.1, 0.2], [0.3, 0.4]])
    assert_equal(layer.weights, expected)


def test_update_biases_correct():
    layer = make_layer_with_gradients()
    old_biases = np.array(layer.biases).copy()
    layer.update_parameters(learning_rate=0.1)
    expected = old_biases - 0.1 * np.array([[0.1], [0.2]])
    assert_equal(layer.biases, expected)


def test_update_weights_with_l2():
    
    # l2 regularization affects weights
    layer = make_layer_with_gradients()
    old_weights = np.array(layer.weights).copy()
    layer.update_parameters(learning_rate=0.1, l2_lambda=0.01)
    expected = old_weights - 0.1 * (np.array([[0.1, 0.2], [0.3, 0.4]]) + 0.01 * old_weights)
    assert_equal(layer.weights, expected)


def test_l2_does_not_affect_biases():
    
    # l2 regularization does not affect biases
    layer = make_layer_with_gradients()
    old_biases = np.array(layer.biases).copy()
    layer.update_parameters(learning_rate=0.1, l2_lambda=0.99)
    expected = old_biases - 0.1 * np.array([[0.1], [0.2]])
    assert_equal(layer.biases, expected)


# ============================================================
# update_parameters — intermediate variables cleared
# ============================================================

def test_intermediates_cleared_after_update():
    
    # clean parameters in the middle
    layer = make_layer_with_gradients()
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
def test_update_wrong_type(learning_rate, l2_lambda):
    layer = make_layer_with_gradients()
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
def test_update_wrong_values(learning_rate, l2_lambda):
    layer = make_layer_with_gradients()
    with pytest.raises(ValueError):
        layer.update_parameters(learning_rate=learning_rate, l2_lambda=l2_lambda)


# ============================================================
# update_parameters — state tests
# ============================================================

def test_update_without_gradients():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1.0, 0.0], [0.0, 1.0]]
    layer.biases  = [[0.0], [0.0]]
    with pytest.raises(ValueError):
        layer.update_parameters(learning_rate=0.1)