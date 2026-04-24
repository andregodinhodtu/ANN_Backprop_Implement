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
    ([[2], [3]], [[6], [3]]),      
    ([[1], [1]], [[3], [1]]),      
    ([[0], [0]], [[1], [0]]),  
])
def test_forward_correct_output_relu(ANN_Layer, input_vector, expected):
    layer = ANN_Layer(
        n=0,
        n_neurons_input=2,
        n_neurons_output=2,
        activation_function="relu"
    )
    layer.weights_matrix = [[1, 1], [0, 1]]
    layer.biases_vector  = [[1], [0]]

    result = layer.forward(input_vector)

    assert np.allclose(result, expected)

@pytest.mark.parametrize("input_vector, expected", [
    ([[2], [3]], [[0], [3]]),     
    ([[1], [1]], [[0], [1]]),     
    ([[6], [3]], [[1], [3]]), 
])
def test_forward_activation_applied_relu(ANN_Layer, input_vector, expected):
    layer = ANN_Layer(
        n=0,
        n_neurons_input=2,
        n_neurons_output=2,
        activation_function="relu"
    )
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
    layer.weights_matrix = [[1, 1], [0, 1]]
    layer.biases_vector  = [[1], [0]]

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
    [[1], [2], "row"],      # row is not a list
    [[1], ["a"]],           # value is not a number
    np.array([["a"], ["b"]]) # numpy array with non-numeric values
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
    [[1], [2], [3]],        # wrong number of rows
])
def test_forward_wrong_values(ANN_Layer, input_vector):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]

    with pytest.raises(ValueError):
        layer.forward(input_vector)

# ============================================================
# Forward — state tests (base Python only)
# ============================================================

def test_forward_weights_not_initialized():
    layer = ANN_Layer_base_python(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    with pytest.raises(ValueError):
        layer.forward([[1], [2]])

def test_forward_biases_not_initialized():
    layer = ANN_Layer_base_python(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    with pytest.raises(ValueError):
        layer.forward([[1], [2]])