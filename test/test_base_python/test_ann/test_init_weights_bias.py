import sys
sys.path.append("src/base_python_version")

import math
import random
import statistics
import pytest
from ANN_layer_base_python import ANN_Layer_base_python as ANN_Layer


# ============================================================
# initialize_weights_bias — happy path
# ============================================================

def test_initialize_weights_shape():
    layer = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer.initialize_weights_bias()
    assert len(layer.weights) == 2
    assert all(len(row) == 3 for row in layer.weights)


def test_initialize_biases_shape():
    layer = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer.initialize_weights_bias()
    assert len(layer.biases) == 2
    assert all(len(row) == 1 for row in layer.biases)


def test_initialize_biases_are_zero():
    layer = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer.initialize_weights_bias()
    for row in layer.biases:
        assert row == [0.0]


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
    layer1.initialize_weights_bias(rng=random.Random(42))
    layer2.initialize_weights_bias(rng=random.Random(42))
    assert layer1.weights == layer2.weights


def test_initialize_different_seed_different_weights():
    layer1 = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer2 = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer1.initialize_weights_bias(rng=random.Random(42))
    layer2.initialize_weights_bias(rng=random.Random(99))
    assert layer1.weights != layer2.weights


# ============================================================
# initialize_weights_bias — He vs Xavier statistical std check
# ============================================================

def _flatten(weights):
    return [v for row in weights for v in row]


def test_initialize_he_std_relu():
    layer = ANN_Layer(n=0, n_neurons_input=100, n_neurons_output=100, activation_function="relu")
    layer.initialize_weights_bias(rng=random.Random(42))
    expected_std = math.sqrt(2 / 100)
    actual_std = statistics.stdev(_flatten(layer.weights))
    assert abs(actual_std - expected_std) < 0.05


def test_initialize_xavier_std_sigmoid():
    layer = ANN_Layer(n=0, n_neurons_input=100, n_neurons_output=100, activation_function="sigmoid")
    layer.initialize_weights_bias(rng=random.Random(42))
    expected_std = math.sqrt(2 / (100 + 100))
    actual_std = statistics.stdev(_flatten(layer.weights))
    assert abs(actual_std - expected_std) < 0.05
