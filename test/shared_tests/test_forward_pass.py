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
# Forward — happy path (relu)
# ============================================================

@pytest.mark.parametrize("input_vector, expected", [
    # normal positive inputs
    ([[2], [3]], [[6], [3]]),       # z = [6, 3] → relu = [6, 3]
    ([[1], [1]], [[3], [1]]),       # z = [3, 1] → relu = [3, 1]
    ([[0], [0]], [[1], [0]]),       # z = [1, 0] → relu = [1, 0]
])
def test_forward_correct_output_relu(ANN_Layer, input_vector, expected):
    layer = ANN_Layer(
        n=0,
        n_neurons_input=2,
        n_neurons_output=2,
        activation_function="relu"
    )
    # weights = [[1, 1],   biases = [[1],
    #            [0, 1]]              [0]]
    layer.weights_matrix = [[1, 1], [0, 1]]
    layer.biases_vector  = [[1], [0]]

    result = layer.forward(input_vector)

    assert np.allclose(result, expected)

@pytest.mark.parametrize("input_vector, expected", [
    # negative z values must be clipped to 0 by relu
    ([[2], [3]], [[0], [3]]),       # z[0] = 2 + 0 - 5 = -3 → relu = 0
    ([[1], [1]], [[0], [1]]),       # z[0] = 1 + 0 - 5 = -4 → relu = 0
    ([[6], [3]], [[1], [3]]),       # z[0] = 6 + 0 - 5 =  1 → relu = 1
])
def test_forward_activation_applied_relu(ANN_Layer, input_vector, expected):
    layer = ANN_Layer(
        n=0,
        n_neurons_input=2,
        n_neurons_output=2,
        activation_function="relu"
    )
    # weights = [[1, 0],   biases = [[-5],
    #            [0, 1]]              [ 0]]
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[-5], [0]]

    result = layer.forward(input_vector)

    assert np.allclose(result, expected)

# ============================================================
# Forward — happy path (sigmoid)
# ============================================================

@pytest.mark.parametrize("input_vector", [
    [[2], [3]],
    [[0], [0]],
    [[1], [1]],
    [[-1], [-1]],
])
def test_forward_correct_output_sigmoid(ANN_Layer, input_vector):
    layer = ANN_Layer(
        n=0,
        n_neurons_input=2,
        n_neurons_output=2,
        activation_function="sigmoid"
    )
    # weights = [[1, 1],   biases = [[1],
    #            [0, 1]]              [0]]
    layer.weights_matrix = [[1, 1], [0, 1]]
    layer.biases_vector  = [[1], [0]]

    # compute expected directly from formula
    z0 = 1 * input_vector[0][0] + 1 * input_vector[1][0] + 1
    z1 = 0 * input_vector[0][0] + 1 * input_vector[1][0] + 0
    expected = [[1 / (1 + np.exp(-z0))],
                [1 / (1 + np.exp(-z1))]]

    result = layer.forward(input_vector)

    assert np.allclose(result, expected)

@pytest.mark.parametrize("input_vector", [
    [[2], [3]],
    [[0], [0]],
    [[1], [1]],
    [[-1], [-1]],
])
def test_forward_sigmoid_output_range(ANN_Layer, input_vector):
    layer = ANN_Layer(
        n=0,
        n_neurons_input=2,
        n_neurons_output=2,
        activation_function="sigmoid"
    )
    # sigmoid always outputs strictly in (0, 1) regardless of input
    layer.weights_matrix = [[1, 1], [0, 1]]
    layer.biases_vector  = [[1], [0]]

    result = layer.forward(input_vector)
    result_array = np.array(result)

    assert np.all(result_array > 0)
    assert np.all(result_array < 1)

# ============================================================
# Forward — output shape
# ============================================================

def test_forward_output_shape(ANN_Layer):
    layer = ANN_Layer(
        n=0,
        n_neurons_input=3,
        n_neurons_output=2,
        activation_function="relu"
    )
    layer.weights_matrix = [[1, 0, 1], [0, 1, 1]]
    layer.biases_vector  = [[0], [0]]

    result = layer.forward([[1], [2], [3]])

    result_array = np.array(result)
    assert result_array.shape == (2, 1)

# ============================================================
# Forward — intermediate values stored
# ============================================================

def test_forward_stores_z_s_and_a_s(ANN_Layer):
    layer = ANN_Layer(
        n=0,
        n_neurons_input=2,
        n_neurons_output=2,
        activation_function="relu"
    )
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]

    layer.forward([[3], [5]])

    # both z_s and a_s must be stored after forward for backpropagation
    assert layer.z_s is not None
    assert layer.a_s is not None
    assert len(layer.z_s) == 2
    assert len(layer.a_s) == 2

# ============================================================
# Forward — TypeError tests
# ============================================================

@pytest.mark.parametrize("input_vector", [
    "not a list",           # string instead of list
    123,                    # int instead of list
    [[1], [2], "row"],      # mixed types — not all rows are lists
])
def test_forward_wrong_type(ANN_Layer, input_vector):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]

    with pytest.raises(TypeError):
        layer.forward(input_vector)

# ============================================================
# Forward — ValueError tests
# ============================================================

@pytest.mark.parametrize("input_vector", [
    [],                     # empty input
    [[1, 2], [3, 4]],       # rows have more than 1 element
    [[1], [2], [3]],        # wrong number of rows (dimension mismatch)
])
def test_forward_wrong_values(ANN_Layer, input_vector):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]

    with pytest.raises(ValueError):
        layer.forward(input_vector)

# ============================================================
# Forward — state tests (weights/biases not initialized)
# ============================================================

def test_forward_weights_not_initialized(ANN_Layer):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(ValueError):
        layer.forward([[1], [2]])

def test_forward_biases_not_initialized(ANN_Layer):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    with pytest.raises(ValueError):
        layer.forward([[1], [2]])