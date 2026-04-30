import sys

sys.path.append("src/numpy_version")

import pytest
import numpy as np
from ANN_layer_numpy import ANN_Layer_numpy as ANN_Layer

# ============================================================
# initialize_weights_bias — happy path
# ============================================================

def test_initialize_weights_shape():
    layer = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer.initialize_weights_bias()
    assert np.array(layer.weights).shape == (2, 3)

def test_initialize_biases_shape():
    layer = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer.initialize_weights_bias()
    assert np.array(layer.biases).shape == (2, 1)

def test_initialize_biases_are_zero():
    layer = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer.initialize_weights_bias()
    assert np.allclose(layer.biases, 0)

def test_initialize_weights_not_none():
    layer = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer.initialize_weights_bias()
    assert layer.weights is not None

# ============================================================
# initialize_weights_bias — seed reproducibility
# ============================================================

def test_initialize_same_seed_same_weights():
    layer1 = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer2 = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer1.initialize_weights_bias(rng=np.random.default_rng(42))
    layer2.initialize_weights_bias(rng=np.random.default_rng(42))
    assert np.allclose(layer1.weights, layer2.weights)

def test_initialize_different_seed_different_weights():
    layer1 = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer2 = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer1.initialize_weights_bias(rng=np.random.default_rng(42))
    layer2.initialize_weights_bias(rng=np.random.default_rng(99))
    assert not np.allclose(layer1.weights, layer2.weights)

# ============================================================
# initialize_weights_bias — He vs Xavier
# ============================================================

def test_initialize_he_std_relu():
    layer = ANN_Layer(n=0, n_neurons_input=100, n_neurons_output=100, activation_function="relu")
    layer.initialize_weights_bias(rng=np.random.default_rng(42))
    expected_std = np.sqrt(2 / 100)
    assert abs(np.std(layer.weights) - expected_std) < 0.05

def test_initialize_xavier_std_sigmoid():
    layer = ANN_Layer(n=0, n_neurons_input=100, n_neurons_output=100, activation_function="sigmoid")
    layer.initialize_weights_bias(rng=np.random.default_rng(42))
    expected_std = np.sqrt(2 / (100 + 100))
    assert abs(np.std(layer.weights) - expected_std) < 0.05
