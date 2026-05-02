import sys
sys.path.append("src/base_python_version")

import math
import pytest
from ANN_layer_base_python import ANN_Layer_base_python as ANN_Layer


# ============================================================
# Helper
# ============================================================

def assert_matrix_equal(actual, expected):
    """Assert two list-of-lists matrices are equal in shape and values."""
    assert len(actual) == len(expected), (
        f"Row count mismatch: got {len(actual)}, expected {len(expected)}"
    )
    for row_a, row_e in zip(actual, expected):
        assert len(row_a) == len(row_e), (
            f"Column count mismatch: got {len(row_a)}, expected {len(row_e)}"
        )
        for a, e in zip(row_a, row_e):
            assert a == pytest.approx(e)


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
    layer.weights = [[1, 1], [0, 1]]
    layer.biases  = [[1], [0]]

    result = layer.forward(input_vector)
    assert_matrix_equal(result, expected)


@pytest.mark.parametrize("input_vector, expected", [
    ([[2], [3]], [[0], [3]]),
    ([[1], [1]], [[0], [1]]),
    ([[6], [3]], [[1], [3]]),
])
def test_forward_activation_applied_relu(input_vector, expected):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[-5], [0]]

    result = layer.forward(input_vector)
    assert_matrix_equal(result, expected)


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
    layer.weights = [[1, 1], [0, 1]]
    layer.biases  = [[1], [0]]

    z0 = 1 * input_vector[0][0] + 1 * input_vector[1][0] + 1
    z1 = 0 * input_vector[0][0] + 1 * input_vector[1][0] + 0
    expected = [[1 / (1 + math.exp(-z0))],
                [1 / (1 + math.exp(-z1))]]

    result = layer.forward(input_vector)
    assert_matrix_equal(result, expected)


@pytest.mark.parametrize("input_vector", [
    [[2], [3]],
    [[0], [0]],
    [[1], [1]],
    [[-1], [-1]],
])
def test_forward_sigmoid_output_range(input_vector):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="sigmoid")
    layer.weights = [[1, 1], [0, 1]]
    layer.biases  = [[1], [0]]

    result = layer.forward(input_vector)
    for row in result:
        for val in row:
            assert 0 < val < 1


# ============================================================
# Forward — output shape
# ============================================================

def test_forward_output_shape():
    layer = ANN_Layer(n=0, n_neurons_input=3, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0, 1], [0, 1, 1]]
    layer.biases  = [[0], [0]]

    result = layer.forward([[1], [2], [3]])
    assert len(result) == 2
    assert all(len(row) == 1 for row in result)


# ============================================================
# Forward — intermediate values stored
# ============================================================

def test_forward_stores_z_s_and_a_s():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]

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
])
def test_forward_wrong_type(input_vector):
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]

    with pytest.raises(TypeError):
        layer.forward(input_vector)


# ============================================================
# Forward — ValueError tests
# ============================================================

def test_forward_wrong_row_count():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]

    with pytest.raises(ValueError):
        layer.forward([[1], [2], [3]])  # 3 rows, expects 2


def test_forward_empty_input():
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]

    with pytest.raises(ValueError):
        layer.forward([])


def test_forward_wrong_column_count():
    """Each row in input_vector must contain exactly 1 element."""
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    layer.weights = [[1, 0], [0, 1]]
    layer.biases  = [[0], [0]]

    with pytest.raises(ValueError):
        layer.forward([[1, 2], [3, 4]])  # multi-column rows
