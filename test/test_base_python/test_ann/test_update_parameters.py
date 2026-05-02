import sys
sys.path.append("src/base_python_version")

import copy
import pytest
from ANN_layer_base_python import ANN_Layer_base_python as ANN_Layer


# ============================================================
# Helper
# ============================================================

def assert_matrix_equal(actual, expected):
    """Assert two list-of-lists matrices are equal in shape and values."""
    assert len(actual) == len(expected)
    for row_a, row_e in zip(actual, expected):
        assert len(row_a) == len(row_e)
        for a, e in zip(row_a, row_e):
            assert a == pytest.approx(e)


# ============================================================
# Helper to set up a layer with gradients ready
# ============================================================

def make_layer_with_gradients():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1.0, 0.0], [0.0, 1.0]]
    layer.biases  = [[0.0], [0.0]]
    layer.forward([[1], [2]])
    layer.compute_activation_derivatives()
    # Manually set gradients (list-of-lists for the base-Python version)
    layer.dweights = [[0.1, 0.2], [0.3, 0.4]]
    layer.dbiases  = [[0.1], [0.2]]
    return layer


# ============================================================
# update_parameters — happy path
# ============================================================

def test_update_weights_correct():
    layer = make_layer_with_gradients()
    old_weights = copy.deepcopy(layer.weights)
    layer.update_parameters(learning_rate=0.1)
    expected = [
        [old_weights[i][j] - 0.1 * [[0.1, 0.2], [0.3, 0.4]][i][j] for j in range(2)]
        for i in range(2)
    ]
    assert_matrix_equal(layer.weights, expected)


def test_update_biases_correct():
    layer = make_layer_with_gradients()
    old_biases = copy.deepcopy(layer.biases)
    layer.update_parameters(learning_rate=0.1)
    expected = [
        [old_biases[i][0] - 0.1 * [[0.1], [0.2]][i][0]]
        for i in range(2)
    ]
    assert_matrix_equal(layer.biases, expected)


def test_update_weights_with_l2():
    layer = make_layer_with_gradients()
    old_weights = copy.deepcopy(layer.weights)
    dweights = [[0.1, 0.2], [0.3, 0.4]]
    layer.update_parameters(learning_rate=0.1, l2_lambda=0.01)
    expected = [
        [old_weights[i][j] - 0.1 * (dweights[i][j] + 0.01 * old_weights[i][j])
         for j in range(2)]
        for i in range(2)
    ]
    assert_matrix_equal(layer.weights, expected)


def test_l2_does_not_affect_biases():
    layer = make_layer_with_gradients()
    old_biases = copy.deepcopy(layer.biases)
    dbiases = [[0.1], [0.2]]
    layer.update_parameters(learning_rate=0.1, l2_lambda=0.99)
    expected = [
        [old_biases[i][0] - 0.1 * dbiases[i][0]]
        for i in range(2)
    ]
    assert_matrix_equal(layer.biases, expected)


# ============================================================
# update_parameters — intermediate variables cleared
# ============================================================

def test_intermediates_cleared_after_update():
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
