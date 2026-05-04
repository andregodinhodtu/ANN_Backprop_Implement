import sys
sys.path.append("src/base_python_version")

import random
import pytest
from ANN_base_python import ANN_base_python as ANN


# ============================================================
# Valid input tests
# ============================================================

def test_input_ANN():
    
    # assignments are correct
    ann = ANN(
        n_layers=3,
        n_neurons_each_layer=[4, 5, 2],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="binarycrossentropy",
    )
    assert ann.n_layers == 3
    assert ann.n_neurons_each_layer == [4, 5, 2]
    assert ann.activation_hidden == "relu"
    assert ann.activation_output == "sigmoid"
    assert ann.loss_function == "binarycrossentropy"


@pytest.mark.parametrize("activation_hidden", ["relu", "sigmoid", "leaky_relu"])
def test_all_hidden_activations(activation_hidden):
    ann = ANN(
        n_layers=2,
        n_neurons_each_layer=[3, 2],
        activation_hidden=activation_hidden,
        activation_output="sigmoid",
        loss_function="binarycrossentropy",
    )
    assert ann.activation_hidden == activation_hidden


@pytest.mark.parametrize("activation_output", ["relu", "sigmoid", "leaky_relu"])
def test_all_output_activations(activation_output):
    
    # Use a loss function that doesn't constrain the output activation
    ann = ANN(
        n_layers=2,
        n_neurons_each_layer=[3, 2],
        activation_hidden="relu",
        activation_output=activation_output,
        loss_function="mse",
    )
    assert ann.activation_output == activation_output


# ============================================================
# Valid input test — initial state
# ============================================================

def test_initial_state():
    ann = ANN(
        n_layers=3,
        n_neurons_each_layer=[4, 5, 2],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="binarycrossentropy",
    )
    assert ann.layers is not None
    # n_layers - 1 connections (4->5, 5->2)
    assert len(ann.layers) == 2
    assert ann.rng is not None


def test_rng_passed_through():
    rng = random.Random(42)
    ann = ANN(
        n_layers=2,
        n_neurons_each_layer=[3, 2],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="binarycrossentropy",
        rng=rng,
    )
    assert ann.rng is rng


# ============================================================
# ValueError tests
# ============================================================

# Error scenarios
@pytest.mark.parametrize(
    "n_layers, n_neurons_each_layer, activation_hidden, activation_output, loss_function",
    [
        (1, [3],          "relu",   "sigmoid", "binarycrossentropy"),  # n_layers < 2
        (0, [],           "relu",   "sigmoid", "binarycrossentropy"),  # n_layers < 2
        (3, [4, 5],       "relu",   "sigmoid", "binarycrossentropy"),  # length mismatch (too few)
        (2, [4, 5, 2],    "relu",   "sigmoid", "binarycrossentropy"),  # length mismatch (too many)
        (2, [4, 0],       "relu",   "sigmoid", "binarycrossentropy"),  # zero layer size
        (2, [4, -1],      "relu",   "sigmoid", "binarycrossentropy"),  # negative layer size
        (2, [3, 2],       "banana", "sigmoid", "binarycrossentropy"),  # unknown hidden activation
        (2, [3, 2],       "relu",   "banana",  "binarycrossentropy"),  # unknown output activation
        (2, [3, 2],       "relu",   "relu",    "binarycrossentropy"),  # bce requires sigmoid output
        (2, [3, 2],       "relu",   "sigmoid", "banana"),              # unknown loss function
    ],
)
def test_wrong_values(n_layers, n_neurons_each_layer, activation_hidden,
                      activation_output, loss_function):
    with pytest.raises(ValueError):
        ANN(
            n_layers=n_layers,
            n_neurons_each_layer=n_neurons_each_layer,
            activation_hidden=activation_hidden,
            activation_output=activation_output,
            loss_function=loss_function,
        )
