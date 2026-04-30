import sys

sys.path.append("src/numpy_version")

import pytest
import numpy as np
from ANN_layer_numpy import ANN_Layer_numpy as ANN_Layer

# ============================================================
# Forward — happy path (relu)
# ============================================================

@pytest.mark.parametrize("input_vector, expected", [
    ([[2], [3]], [[6], [3]]),
    ([[1], [1]], [[3], [1]]),
    ([[0], [0]], [[1], [0]]),
])
def test_forward_correct_output_relu(input_vector, expected):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 1], [0, 1]]
    layer.biases_vector  = [[1], [0]]

    result = layer.forward(input_vector)
    assert np.allclose(result, expected)

@pytest.mark.parametrize("input_vector, expected", [
    ([[2], [3]], [[0], [3]]),
    ([[1], [1]], [[0], [1]]),
    ([[6], [3]], [[1], [3]]),
])
def test_forward_activation_applied_relu(input_vector, expected):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
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
def test_forward_correct_output_sigmoid(input_vector):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="sigmoid")
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
def test_forward_sigmoid_output_range(input_vector):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="sigmoid")
    layer.weights_matrix = [[1, 1], [0, 1]]
    layer.biases_vector  = [[1], [0]]

    result = layer.forward(input_vector)
    result_array = np.array(result)
    assert np.all(result_array > 0)
    assert np.all(result_array < 1)

# ============================================================
# Forward — output shape
# ============================================================

def test_forward_output_shape():
    layer = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0, 1], [0, 1, 1]]
    layer.biases_vector  = [[0], [0]]

    result = layer.forward([[1], [2], [3]])
    result_array = np.array(result)
    assert result_array.shape == (2, 1)

# ============================================================
# Forward — intermediate values stored
# ============================================================

def test_forward_stores_z_s_and_a_s():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
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
    "not a list",            # string instead of list
    123,                     # int instead of list
    [[1], [2], "row"],       # row is not a list
    [[1], ["a"]],            # value is not a number
    np.array([["a"], ["b"]]) # numpy array with non-numeric values
])
def test_forward_wrong_type(input_vector):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]

    with pytest.raises(TypeError):
        layer.forward(input_vector)

# ============================================================
# Forward — ValueError tests
# ============================================================

def test_forward_wrong_row_count():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]

    with pytest.raises(ValueError):
        layer.forward([[1], [2], [3]])  # 3 rows, expects 2

def test_forward_empty_input():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]

    with pytest.raises(ValueError):
        layer.forward([])

# ============================================================
# Forward — NumPy vectorization tests
# ============================================================

def test_forward_accepts_batch():
    """NumPy treats multi-column input as a batch of samples."""
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[1, 0], [0, 1]]
    layer.biases_vector  = [[0], [0]]

    # Batch of 2 samples (each column is one sample)
    input_batch = np.array([[1, 3],
                            [2, 4]])
    out = layer.forward(input_batch)

    expected = np.array([[1, 3], [2, 4]])
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, expected)

def test_forward_batch_matches_individual_calls():
    """Batched forward must produce the same result as sample-by-sample calls."""
    layer = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer.weights_matrix = [[0.5, -1.0, 0.2], [1.0, 0.3, -0.4]]
    layer.biases_vector  = [[0.1], [-0.2]]

    samples = [
        np.array([[1.0], [2.0], [3.0]]),
        np.array([[-1.0], [0.5], [2.0]]),
        np.array([[0.0], [0.0], [0.0]]),
        np.array([[10.0], [-5.0], [3.0]]),
    ]

    batch = np.hstack(samples)
    batched_output = layer.forward(batch)
    individual_outputs = np.hstack([layer.forward(s) for s in samples])

    np.testing.assert_allclose(batched_output, individual_outputs, atol=1e-10)
