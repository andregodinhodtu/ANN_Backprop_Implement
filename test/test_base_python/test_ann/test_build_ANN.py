import sys
sys.path.append("src/base_python_version")
import pytest
from ANN_base_python import ANN_base_python as ANN
from ANN_layer_base_python import ANN_Layer_base_python


# ============================================================
# _build_ANN — happy path
# ============================================================
def test_build_standard_ANN():
    ann = ANN(
        n_layers=3,
        n_neurons_each_layer=[4, 5, 2],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="binarycrossentropy",
    )

    # 3 layers in the architecture => 2 weight layers between them
    assert len(ann.layers) == 2

    # Verify first layer (input -> hidden)
    layer_0_to_1 = ann.layers[0]
    assert layer_0_to_1.n_neurons_input == 4
    assert layer_0_to_1.n_neurons_output == 5
    assert layer_0_to_1.n == 1  # n=i+1 with i=0
    assert layer_0_to_1.activation_function == "relu"

    # Verify second layer (hidden -> output)
    layer_1_to_2 = ann.layers[1]
    assert layer_1_to_2.n_neurons_input == 5
    assert layer_1_to_2.n_neurons_output == 2
    assert layer_1_to_2.n == 2  # n=i+1 with i=1
    assert layer_1_to_2.activation_function == "sigmoid"


def test_build_deep_ANN():
    """Deeper net, make sure layer count and shapes propagate correctly."""
    ann = ANN(
        n_layers=5,
        n_neurons_each_layer=[10, 8, 6, 4, 2],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="binarycrossentropy",
    )
    assert len(ann.layers) == 4

    expected_shapes = [(10, 8), (8, 6), (6, 4), (4, 2)]
    for layer, (n_in, n_out) in zip(ann.layers, expected_shapes):
        assert layer.n_neurons_input == n_in
        assert layer.n_neurons_output == n_out

    # Only the last layer uses the output activation
    assert ann.layers[-1].activation_function == "sigmoid"
    for layer in ann.layers[:-1]:
        assert layer.activation_function == "relu"


# ============================================================
# weights setter — TypeError
# ============================================================
def test_weights_setter_wrong_type():
    """Setting weights to a non-list should raise TypeError."""
    layer = ANN_Layer_base_python(
        n=1,
        n_neurons_input=4,
        n_neurons_output=5,
        activation_function="relu",
    )

    with pytest.raises(TypeError):
        layer.weights = "not a matrix"

    with pytest.raises(TypeError):
        layer.weights = 42

    with pytest.raises(TypeError):
        layer.weights = [1, 2, 3]   # flat list, not list of lists


# ============================================================
# weights setter — ValueError
# ============================================================
def test_weights_setter_wrong_shape():
    """Setting weights to a list-of-lists of the wrong shape should raise ValueError."""
    layer = ANN_Layer_base_python(
        n=1,
        n_neurons_input=4,
        n_neurons_output=5,
        activation_function="relu",
    )

    # Expected shape is (n_neurons_output, n_neurons_input) == (5, 4).
    with pytest.raises(ValueError):
        layer.weights = [[0.0] * 5 for _ in range(4)]   # transposed (4, 5)

    with pytest.raises(ValueError):
        layer.weights = [[0.0] * 5 for _ in range(5)]   # wrong input dim (5, 5)

    with pytest.raises(ValueError):
        layer.weights = [[0.0] * 4 for _ in range(3)]   # wrong output dim (3, 4)
