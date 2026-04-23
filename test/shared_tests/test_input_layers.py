import pytest
import sys
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
# Valid input test
# ============================================================

def test_input_layer(ANN_Layer):
    layer = ANN_Layer(
        n=0,
        n_neurons_input=5,
        n_neurons_output=3,
        activation_function="relu"
    )
    assert layer.n == 0
    assert layer.n_neurons_input == 5
    assert layer.n_neurons_output == 3
    assert layer.activation_function == "relu"

# ============================================================
# TypeError tests
# ============================================================

@pytest.mark.parametrize("n, n_neurons_input, n_neurons_output, activation_function", [
    ("0", 5,   3,   "relu"),   # n wrong type
    (0,   "5", 3,   "relu"),   # n_neurons_input wrong type
    (0,   5,   "3", "relu"),   # n_neurons_output wrong type
    (0,   5,   3,   123),      # activation_function wrong type
])
def test_wrong_types(ANN_Layer, n, n_neurons_input, n_neurons_output, activation_function):
    with pytest.raises(TypeError):
        ANN_Layer(
            n=n,
            n_neurons_input=n_neurons_input,
            n_neurons_output=n_neurons_output,
            activation_function=activation_function
        )

# ============================================================
# ValueError tests
# ============================================================

@pytest.mark.parametrize("n, n_neurons_input, n_neurons_output, activation_function", [
    (-1, 5,  3,  "relu"),     # n negative
    (0,  0,  3,  "relu"),     # n_neurons_input zero
    (0,  -1, 3,  "relu"),     # n_neurons_input negative
    (0,  5,  0,  "relu"),     # n_neurons_output zero
    (0,  5,  -1, "relu"),     # n_neurons_output negative
    (0,  5,  3,  "banana"),   # unknown activation function
])
def test_wrong_values(ANN_Layer, n, n_neurons_input, n_neurons_output, activation_function):
    with pytest.raises(ValueError):
        ANN_Layer(
            n=n,
            n_neurons_input=n_neurons_input,
            n_neurons_output=n_neurons_output,
            activation_function=activation_function
        )