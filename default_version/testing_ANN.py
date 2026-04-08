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


#################################################################
############################## Main #############################
#################################################################

# Run the test
if __name__ == "__main__":
    test_compute_deltas_fixed_params()
    