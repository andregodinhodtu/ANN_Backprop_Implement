import random
from ANN import ANN

#################################################################
############################ Testing ############################
#################################################################

def test_build_ANN():
    # Define a network with 4 layers (including input)
    n_layers = 5
    n_neurons_each_layer = [3, 5, 4, 2, 8]  # 3 inputs, then 5, 4, 2 neurons
    activation_hidden = "relu"
    activation_output = "sigmoid"
    
    # Instantiate ANN
    my_ann = ANN(
        n_layers=n_layers,
        n_neurons_each_layer=n_neurons_each_layer,
        activation_hidden=activation_hidden,
        activation_output=activation_output,
        loss_function = "MSE"
    )
    
    # Print info about each layer
    print("Network built successfully! Layer details:")
    for layer in my_ann.layers:
        print(f"Layer {layer.n}:")
        print(f"  Inputs: {layer.n_neurons_input}")
        print(f"  Outputs: {layer.n_neurons_output}")
        print(f"  Activation: {layer.activation_function}")
        print("  Weights shape:", layer.shape("weights"))
        print("  Biases shape:", layer.shape("biases"))
        print("  Output shape:", layer.shape("output"))
        print()
        
def test_prediction():
    # Define a small ANN with 3 layers: 2 inputs -> 3 hidden -> 1 output
    n_inputs = 2
    n_neurons_each_layer = [n_inputs, 3, 2, 6, 2]
    my_ann = ANN(
        n_layers=len(n_neurons_each_layer),
        n_neurons_each_layer=n_neurons_each_layer,
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="MSE"
    )

    # Initialize weights and biases (optional seed for reproducibility)
    for i, layer in enumerate(my_ann.layers):
        layer.initialize_weights_bias(seed=42)

    # Create a test input vector (column vector format)
    input_vector = [[0.5], [0.2]]

    # Make prediction
    output = my_ann.prediction(input_vector)

    # Print results
    print("Input vector:")
    for row in input_vector:
        print(row)
    print("\nOutput of the ANN:")
    for row in output:
        print(row)

    # Optionally print intermediate layer activations
    for layer in my_ann.layers:
        print(f"\nLayer {layer.n} activations (a_s):")
        for row in layer.a_s:
            print(row)
        print(f"Layer {layer.n} pre-activations (z_s):")
        for row in layer.z_s:
            print(row)
            
            
def test_compute_deltas_fixed_params():
    from ANN import ANN, ANN_Layer  # adjust import if needed

    # Simple network: 2 input, 2 hidden, 1 output
    n_layers = 3
    n_neurons_each_layer = [2, 2, 1]

    # Create ANN
    my_ann = ANN(
        n_layers=n_layers,
        n_neurons_each_layer=n_neurons_each_layer,
        activation_hidden="relu",
        activation_output="relu",
        loss_function="MSE"
    )

    # Manually set weights and biases for reproducible computation
    # Layer 0 (input → hidden)
    my_ann.layers[0].weights = [
        [0.1, 0.2],  # neuron 0 in hidden layer
        [0.3, 0.4]   # neuron 1 in hidden layer
    ]
    my_ann.layers[0].biases = [
        [0.1],
        [0.1]
    ]

    # Layer 1 (hidden → output)
    my_ann.layers[1].weights = [
        [0.5, 0.6]   # single output neuron
    ]
    my_ann.layers[1].biases = [
        [0.2]
    ]

    # Example input (2x1 column vector)
    x = [[0.5], [0.1]]
    # Example target output (1x1 column vector)
    y = [[1.0]]

    # Forward pass
    output = my_ann.prediction(x)
    print("Output of network:", output)

    # Compute deltas
    my_ann._compute_deltas(y)

    # Print deltas for each layer
    for i, layer in enumerate(my_ann.layers):
        print(f"Layer {i} deltas: {layer.delta}")
        
        
def test_activation_derivatives():
    from ANN_layer import ANN_Layer  # adjust import if needed

    # Create a simple layer: 2 inputs, 3 neurons, sigmoid activation
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=3, activation_function="sigmoid")
    
    # Initialize weights and biases for reproducibility
    layer.initialize_weights_bias(seed=42)
    
    # Example input: 2x1 column vector
    input_vector = [[0.5], [0.1]]
    
    # Forward pass
    layer.forward(input_vector)
    
    print("Pre-activations z_s:")
    for i, z in enumerate(layer.z_s):
        print(f"Neuron {i}: {z[0]:.6f}")
    
    # Compute activation derivatives
    derivatives = layer.compute_activation_derivatives()
    
    print("\nActivation derivatives (da/dz):")
    for i, d in enumerate(derivatives):
        print(f"Neuron {i}: {d:.6f}")
        
def test_compute_gradients():
    """
    Test the correctness of compute_gradients using numerical approximation.
    """

    # Create small network
    ann = ANN(
        n_layers=2,
        n_neurons_each_layer=[2, 1],  # 2 inputs → 1 output
        activation_hidden="sigmoid",
        activation_output="sigmoid",
        loss_function="MSE"
    )

    # Manually set weights for reproducibility
    ann.layers[0].weights = [[0.5, -0.3]]
    ann.layers[0].biases = [[0.1]]

    # Define input and target
    x = [[1.0], [2.0]]
    y = [[0.5]]

    # Compute analytical gradients
    ann.compute_gradients(x, y)
    analytical_grad = ann.layers[0].dweights[0][0]

    # Compute numerical gradient (finite difference)
    epsilon = 1e-5

    # Save original weight
    original_weight = ann.layers[0].weights[0][0]

    # L(w + ε)
    ann.layers[0].weights[0][0] = original_weight + epsilon
    out_plus = ann.prediction(x)
    loss_plus = ANN.LOSS_FUNCTIONS["MSE"]["func"](out_plus, y)

    # L(w - ε)
    ann.layers[0].weights[0][0] = original_weight - epsilon
    out_minus = ann.prediction(x)
    loss_minus = ANN.LOSS_FUNCTIONS["MSE"]["func"](out_minus, y)

    # Restore original weight
    ann.layers[0].weights[0][0] = original_weight

    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

    # Compare
    print("Analytical gradient:", analytical_grad)
    print("Numerical gradient :", numerical_grad)
    print("Difference         :", abs(analytical_grad - numerical_grad))
    

def test_relu_gradient_step():
    """
    Test a single gradient descent step on a small ReLU network.
    Prints every step to verify correctness.
    """

    # Small network: 2 → 2 → 1
    ann = ANN(
        n_layers=3,
        n_neurons_each_layer=[2, 2, 1],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="MSE"
    )

    # Manually set weights for deterministic behavior
    ann.layers[0].weights = [
        [0.2, -0.4],
        [0.7, 0.1]
    ]
    ann.layers[0].biases = [[0.0], [0.0]]

    ann.layers[1].weights = [
        [0.6, -0.1]
    ]
    ann.layers[1].biases = [[0.0]]

    # Input and target
    x = [[1.0], [2.0]]
    y = [[1.0]]

    # Forward pass BEFORE update
    print("\n--- FORWARD PASS (BEFORE) ---")
    output_before = ann.prediction(x)
    print("Output:", output_before)

    loss_before = ANN.LOSS_FUNCTIONS["MSE"]["func"](output_before, y)
    print("Loss:", loss_before)

    # Compute gradients
    ann.compute_gradients_sample(x, y)

    print("\n--- GRADIENTS ---")
    for i, layer in enumerate(ann.layers):
        print(f"\nLayer {i +1}")
        print("Deltas:", layer.delta)
        print("dWeights:")
        for row in layer.dweights:
            print(row)
        print("dBiases:", layer.dbiases)

    # Gradient descent step
    lr = 0.1

    for layer in ann.layers:
        for i in range(layer.n_neurons_output):
            for j in range(len(layer.weights[i])):
                layer.weights[i][j] -= lr * layer.dweights[i][j]
            layer.biases[i][0] -= lr * layer.dbiases[i]

    # Forward pass AFTER update
    print("\n--- FORWARD PASS (AFTER) ---")
    output_after = ann.prediction(x)
    print("Output:", output_after)

    loss_after = ANN.LOSS_FUNCTIONS["MSE"]["func"](output_after, y)
    print("Loss:", loss_after)

    # Check improvement
    print("\n--- CHECK ---")
    print("Loss before:", loss_before)
    print("Loss after :", loss_after)

    if loss_after < loss_before:
        print("✅ SUCCESS: Loss decreased")
    else:
        print("❌ WARNING: Loss did NOT decrease")
        
def test_batch_gradient_step():
    """
    Test a single gradient descent step on a small ReLU network using batch gradients.
    Prints every step to verify correctness.
    """
    
    # Small network: 2 → 2 → 1
    ann = ANN(
        n_layers=3,
        n_neurons_each_layer=[2, 2, 1],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="MSE"
    )
    
    # Manually set weights for deterministic behavior
    ann.layers[0].weights = [
        [0.2, -0.4],
        [0.7, 0.1]
    ]
    ann.layers[0].biases = [[0.0], [0.0]]

    ann.layers[1].weights = [
        [0.6, -0.1]
    ]
    ann.layers[1].biases = [[0.0]]
    
    # Batch input and targets
    X_batch = [
        [[1.0], [2.0]],
        [[0.5], [-1.0]]
    ]
    Y_batch = [
        [[1.0]],
        [[0.0]]
    ]
    
    # Forward pass BEFORE update
    print("\n--- FORWARD PASS (BEFORE) ---")
    for x, y in zip(X_batch, Y_batch):
        output = ann.prediction(x)
        print(f"Input: {[xi[0] for xi in x]} -> Output: {output[0][0]}, Target: {y[0][0]}")
    
    # Compute average batch loss
    loss_before = sum(ANN.LOSS_FUNCTIONS["MSE"]["func"](ann.prediction(x), y) for x, y in zip(X_batch, Y_batch))
    print("Average Loss:", loss_before / len(X_batch))
    
    # Compute batch gradients
    ann.compute_gradients_batch(X_batch, Y_batch)
    
    print("\n--- BATCH GRADIENTS ---")
    for i, layer in enumerate(ann.layers):
        print(f"\nLayer {i + 1}")
        print("dWeights:")
        for row in layer.dweights:
            print(row)
        print("dBiases:", layer.dbiases)
    
    # Gradient descent step
    lr = 0.1
    for layer in ann.layers:
        for i in range(layer.n_neurons_output):
            for j in range(len(layer.weights[i])):
                layer.weights[i][j] -= lr * layer.dweights[i][j]
            layer.biases[i][0] -= lr * layer.dbiases[i]
    
    # Forward pass AFTER update
    print("\n--- FORWARD PASS (AFTER) ---")
    for x, y in zip(X_batch, Y_batch):
        output = ann.prediction(x)
        print(f"Input: {[xi[0] for xi in x]} -> Output: {output[0][0]}, Target: {y[0][0]}")
    
    # Compute average batch loss after update
    loss_after = sum(ANN.LOSS_FUNCTIONS["MSE"]["func"](ann.prediction(x), y) for x, y in zip(X_batch, Y_batch))
    print("Average Loss:", loss_after / len(X_batch))
    
    # Check improvement
    print("\n--- CHECK ---")
    if loss_after < loss_before:
        print("✅ SUCCESS: Loss decreased after batch gradient step")
    else:
        print("❌ WARNING: Loss did NOT decrease after batch gradient step")
        
        
def test_layer_update_parameters():
    """
    Test the update_parameters method of ANN_Layer.
    - Small network: 2 → 2 → 1
    - Uses ReLU hidden layer and sigmoid output
    - Prints everything step by step to verify
    """

    # --- Create small ANN ---
    ann = ANN(
        n_layers=3,
        n_neurons_each_layer=[2, 2, 1],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="MSE"
    )

    # --- Set deterministic weights and biases ---
    ann.layers[0].weights = [[0.2, -0.4], [0.7, 0.1]]
    ann.layers[0].biases = [[0.0], [0.0]]

    ann.layers[1].weights = [[0.6, -0.1]]
    ann.layers[1].biases = [[0.0]]

    # --- Input and target ---
    x = [[1.0], [2.0]]
    y = [[1.0]]

    # --- Forward pass before update ---
    print("\n--- FORWARD PASS BEFORE UPDATE ---")
    output_before = ann.prediction(x)
    print("Output:", output_before)
    loss_before = ANN.LOSS_FUNCTIONS["MSE"]["func"](output_before, y)
    print("Loss:", loss_before)

    # --- Compute gradients for single sample ---
    ann.compute_gradients_sample(x, y)

    # --- Print gradients ---
    print("\n--- GRADIENTS ---")
    for i, layer in enumerate(ann.layers):
        print(f"\nLayer {i+1}")
        print("dWeights:", layer.dweights)
        print("dBiases:", layer.dbiases)
        print("Delta:", layer.delta)

    # --- 7️⃣ Update parameters ---
    lr = 0.1
    for layer in ann.layers:
        layer.update_parameters(lr)

    # --- Check that z_s, a_s, and gradients are cleared ---
    print("\n--- CHECK CLEANUP ---")
    for i, layer in enumerate(ann.layers):
        print(f"Layer {i+1}: z_s={layer.z_s}, a_s={layer.a_s}, dweights={layer.dweights}, dbiases={layer.dbiases}")

    # --- Forward pass after update ---
    print("\n--- FORWARD PASS AFTER UPDATE ---")
    output_after = ann.prediction(x)
    print("Output:", output_after)
    loss_after = ANN.LOSS_FUNCTIONS["MSE"]["func"](output_after, y)
    print("Loss:", loss_after)

    # --- Verify improvement ---
    print("\n--- LOSS IMPROVEMENT ---")
    if loss_after < loss_before:
        print("✅ SUCCESS: Loss decreased")
    else:
        print("❌ WARNING: Loss did NOT decrease")

def test_train_function():
    """
    Test the ANN training with a small network and a tiny dataset.
    Prints loss per epoch and shows weight/bias updates.
    """

    # Create a small network: 2 → 2 → 1
    ann = ANN(
        n_layers=3,
        n_neurons_each_layer=[2, 2, 1],
        activation_hidden="relu",
        activation_output="sigmoid",
        loss_function="MSE"
    )

    # Manually set weights and biases for deterministic behavior
    ann.layers[0].weights = [
        [0.2, -0.4],
        [0.7, 0.1]
    ]
    ann.layers[0].biases = [[0.0], [0.0]]

    ann.layers[1].weights = [
        [0.6, -0.1]
    ]
    ann.layers[1].biases = [[0.0]]

    # Toy dataset
    X = [
        [[1.0], [2.0]],
        [[0.5], [-1.0]],
        [[-1.0], [1.5]]
    ]
    Y = [
        [[1.0]],
        [[0.0]],
        [[1.0]]
    ]

    # Train with batch_size=1 to see individual steps
    print("\n--- TRAINING START ---\n")
    ann.train(X, Y, epochs=10, learning_rate=0.1, batch_size=1, verbose=True)

    # Print final weights and biases
    print("\n--- FINAL WEIGHTS AND BIASES ---")
    for i, layer in enumerate(ann.layers):
        print(f"\nLayer {i+1}:")
        layer.print_weights_and_biases()

#################################################################
############################## Main #############################
#################################################################

# Run the test
if __name__ == "__main__":
    
    test_train_function()
    
    """# Simple network: 2 input, 2 hidden, 1 output
    n_layers = 4
    n_neurons_each_layer = [2, 5,8, 1]

    # Create ANN
    my_ann = ANN(
        n_layers=n_layers,
        n_neurons_each_layer=n_neurons_each_layer,
        activation_hidden="sigmoid",
        activation_output="sigmoid",
        loss_function="MSE"
    )
    
    print("Network built successfully! Layer details:")
    for layer in my_ann.layers:
        print(f"Layer {layer.n}:")
        print(f"  Inputs: {layer.n_neurons_input}")
        print(f"  Outputs: {layer.n_neurons_output}")
        print(f"  Activation: {layer.activation_function}")
        print("  Weights shape:", layer.shape("weights"))
        print("  Biases shape:", layer.shape("biases"))
        print("  Output shape:", layer.shape("output"))
        print()
    
    input_vector = [[1], [2]]
    y = [[1]]
    my_ann.prediction(input_vector)
    my_ann._compute_deltas(y)"""
    