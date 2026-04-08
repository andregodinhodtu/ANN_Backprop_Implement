import random
from ANN_layer import ANN_Layer

#################################################################
############################ Testing ############################
#################################################################

def test_initialization_weights_in_layer():
    """
    Test the initialization of weights and biases in a single ANN layer.
    This function:
    - Creates a layer
    - Initializes weights and biases with a fixed seed for reproducibility
    - Prints the weights and biases
    - Checks error handling for invalid inputs
    """

    print("=== Testing valid layer initialization ===")
    try:
        test_ANN = ANN_Layer(
            n=0,
            n_neurons_input=3,
            n_neurons_output=4,
            activation_function="relu"
        )

        # Initialize weights and biases with a fixed seed
        test_ANN.initialize_weights_bias(seed=42)

        # Print the initialized weights and biases
        test_ANN.print_weights_and_biases()
    except Exception as e:
        print("Unexpected error during valid initialization:", e)

    print("\n=== Testing invalid inputs ===")
    # Edge case: zero neurons input
    try:
        bad_layer = ANN_Layer(
            n=1,
            n_neurons_input=0,
            n_neurons_output=4,
            activation_function="relu"
        )
    except ValueError as e:
        print("Caught expected ValueError:", e)

    # Edge case: unknown activation function
    try:
        bad_layer2 = ANN_Layer(
            n=1,
            n_neurons_input=3,
            n_neurons_output=4,
            activation_function="unknown_activation"
        )
    except ValueError as e:
        print("Caught expected ValueError:", e)
        
def test_product_full():
    """
    Test multiple matrix x vector examples with a single ANN layer.
    Verifies output for valid inputs and catches errors for invalid sizes.
    """

    # Create a layer: 3 inputs -> 4 neurons
    test_ANN = ANN_Layer(
        n=0,
        n_neurons_input=3,
        n_neurons_output=4,
        activation_function="relu"
    )

    # Manually set integer weights
    test_ANN.weights = [
        [1, 0, 2],
        [0, 1, -1],
        [2, 2, 2],
        [-1, 0, 1]
    ]

    # Biases are ignored
    test_ANN.biases = [[0], [0], [0], [0]]

    # Valid test cases: (input_vector, expected_output)
    valid_cases = [
        (
            [[1], [2], [3]],  # input
            [[7], [-1], [12], [2]]  # expected output
        ),
        (
            [[0], [0], [0]],
            [[0], [0], [0], [0]]
        )
    ]

    # Invalid test cases: input_vectors with wrong sizes
    invalid_cases = [
        [[1], [1]],           # 2x1 instead of 3x1
        [[1], [2], [3], [4]]  # 4x1 instead of 3x1
    ]

    # --- Test valid cases ---
    for idx, (vec, expected) in enumerate(valid_cases):
        try:
            print(f"\nValid Test {idx+1}: input_vector = {vec}")
            output = test_ANN * vec
            if output == expected:
                print("PASS ✅ Output matches expected:", output)
            else:
                print("FAIL ❌ Output does not match expected")
                print("Expected:", expected)
                print("Got     :", output)
        except Exception as e:
            print("ERROR ❌ during valid test:", e)

    # --- Test invalid cases ---
    for idx, vec in enumerate(invalid_cases):
        try:
            print(f"\nInvalid Test {idx+1}: input_vector = {vec}")
            output = test_ANN * vec
            print("FAIL ❌ Expected error but got output:", output)
        except Exception as e:
            print("PASS ✅ Correctly caught error:", e)
            

def test_layer_forward():
    """
    Test the forward pass of a single ANN layer using __mul__ and __add__.
    Includes:
    - Multiplication (weights * input_vector)
    - Adding biases
    - Combined usage to produce final pre-activation output
    """

    # Create a layer: 3 inputs -> 4 neurons
    layer = ANN_Layer(
        n=0,
        n_neurons_input=3,
        n_neurons_output=4,
        activation_function="relu"
    )

    # Manually set integer weights
    layer.weights = [
        [1, 0, 2],  # neuron 0
        [0, 1, -1], # neuron 1
        [2, 2, 2],  # neuron 2
        [-1, 0, 1]  # neuron 3
    ]

    # Set biases
    layer.biases = [[1], [2], [3], [4]]

    # Input vector
    input_vector = [[1], [2], [3]]

    # --- Step 1: Multiply weights by input ---
    mul_output = layer * input_vector
    print("Multiplication (W*x) output:", mul_output)
    # Expected: [[7], [-1], [12], [2]]

    # --- Step 2: Add biases using __add__ ---
    final_output = layer + mul_output 
    print("After adding biases (W*x + b):", final_output)
    # Expected: [[8], [1], [15], [6]]

    # --- Step 3: Test combined one-liner ---
    combined_output = layer + (layer * input_vector)
    print("Combined one-liner output:", combined_output)
    # Should match final_output
    assert combined_output == final_output, "Combined output does not match step-by-step output!"

    print("\nAll forward pass tests passed ✅")
    
def test_apply_activation():
    """
    Test the _apply_activation method of a layer.
    Uses a simple ReLU activation and a fixed z_s matrix.
    """
    
    # Define a simple layer class for testing
    class TestLayer:
        def __init__(self):
            self.activation_function = lambda x: max(0, x)  # ReLU
            self.z_s = [[-1], [0], [3], [5]]  # pre-activation outputs

        _apply_activation = ANN_Layer._apply_activation  # use your method
    
    # Instantiate layer
    layer = TestLayer()
    
    # Apply activation
    layer._apply_activation()
    
    print("Pre-activation z_s:", layer.z_s)
    print("Activated output:", layer.a_s)
    
    # Check expected values
    expected = [[0], [0], [3], [5]]
    assert layer.a_s == expected, "Activation output does not match expected values!"
    
    print("\n_apply_activation test passed ✅")
    
def test_layer_call():
    """
    Test the __call__ method of ANN_Layer.
    Verifies that the forward pass + activation works correctly.
    """
    # Create a simple layer with 2 inputs and 2 outputs
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    
    # Manually set weights and biases for predictable output
    layer.weights = [
        [1, 2],  # neuron 1
        [3, 4]   # neuron 2
    ]
    layer.biases = [
        [1],     # bias for neuron 1
        [-2]     # bias for neuron 2
    ]
    
    # Define a test input vector (column vector)
    input_vector = [
        [1],  # input 1
        [2]   # input 2
    ]
    
    # Call the layer using the __call__ method
    output = layer(input_vector)
    
    # Expected calculation:
    # Neuron 1: max(0, (1*1 + 2*2) + 1) = max(0, 1+4+1) = 6
    # Neuron 2: max(0, (3*1 + 4*2) + (-2)) = max(0, 3+8-2) = 9
    expected_output = [[6], [9]]
    
    print("Output from layer __call__:", output)
    print("Expected output:", expected_output)
    
    assert output == expected_output, "Test failed: output does not match expected result."
    print("\nTest passed ✅")


#################################################################
############################## Main #############################
#################################################################

# Run the test
if __name__ == "__main__":
    test_layer_call()
    