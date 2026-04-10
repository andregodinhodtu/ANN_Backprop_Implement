import numpy as np
import random

# no seed - each ANN should be initialised with random weights + epochs must differ 

class ANN_Layer():
    
    # sigmoid for binary classification problem
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(a):
        return a * (1.0 - a)
    
    # ReLu for hidden layers
    @staticmethod
    def relu(x):
        return np.maximum(0,x)

    @staticmethod
    def relu_deriv(a):
        return (a > 0).astype(float) # returns 1.0 if a > 0 and 0.0 otherwise
    


    #------------------------------------------------------------------------------


    def __init__(self, n, n_neurons_input, n_neurons_output, activation_function="relu"):
        """
        Initialize a layer in the neural network.

        Parameters:
        -----------
        n : int
            Layer index.
    
        n_neurons_input : int
            Number of input neurons coming into this layer
            (size of the vector received from previous layer or raw input).

        n_neuorns_output : int
            Number of neurons in this layer (after activation).

        activation_function : str
            Activation function to apply to this layer's output.
            Default: "relu"
            Other option: "sigmoid"
        """

        # Type and value checks
        if not isinstance(n, int):
            raise TypeError("n must be an Integer")
        if not isinstance(n_neurons_input, int):
            raise TypeError("n_neurons_input must be an Integer")
        if not isinstance(n_neurons_output, int):
            raise TypeError("n_neurons_output must be an Integer")
        if not isinstance(activation_function, str):
            raise TypeError("activation_function should be a String")
        
        if n < 0:
            raise ValueError("n must be >= 0")
        if n_neurons_input <= 0:
            raise ValueError("n_neurons_input must be > 0")
        if n_neurons_output <= 0:
            raise ValueError("n_neurons_output must be > 0") 
        if activation_function not in ["relu", "sigmoid"]:
            raise ValueError(f"Unknown hidden activation function: {activation_function}")

        # Layer structure
        self.n = n
        self.n_neurons_input = n_neurons_input
        self.n_neurons_output = n_neurons_output
        self.activation_function = activation_function

        # Parameters
        # - weights: (n_neuorns_output, n_neurons_input)
        # - bias: (n_neuorns_output, 1)
        self.weights = None
        self.biases = None
        
        # Intermediate values
        self.z_s = None
        self.a_s = None
        
        # Backpropagation
        self.activation_derivatives = None
        self.delta = None  # to store error signal for backprop
        
    
    def __call__(self, input_vector):
        """
        Enables calling the layer like a function: layer(input_vector).

        Performs the forward pass and returns the activated output.

        Parameters:
        -----------
        input_vector : list of lists
            Input to the layer (column vector or batch of column vectors).

        """
        return self.forward(input_vector)           # compute pre-activation z_s + activation a_s
        
        
            
    def initialize_weights_bias(self, seed=None):
        """
        Initialize weights randomly in a fixed range [-0.5, 0.5].
        Biases are initialized to 0.
    
        Parameters:
        -----------
        seed : int or None
            Optional seed for the random number generator to make results reproducible.
        """
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility

        # Weight matrix: shape (n_neurons_output, n_neurons_input)
        self.weights = np.random.unifrom(-0.5, 0.5, (self.n_neurons_output, self.n_neurons_input))

        # Bias vector: shape (n_neurons_output, 1)
        self.biases = np.zeros(self.n_neurons_output, 1)


    
    def forward(self, input_vector): # fix to already apply activation
        
        # ensure input is a 2D np array
        x = np.array(input_vector)
        if x.ndim == 1:
            x = x.reshape(-1,1) # column vector (n_features, 1)
        
        # fill in z_s: z = w * x + b
        self.z_s = np.dot(self.weights, x) + self.biases

        # apply activation for a_s
        if self.activation_function == "relu":
            self.a_s = self.relu(self.z_s)
        elif self.activation_function == "sigmoid":
            self.a_s = self.sigmoid(self.z_s)
        else:
            raise ValueError("Unknown activation function")
        
        return self.a_s
    


    def compute_activation_derivatives(self):
        """
        Compute the derivative of the activation function for each neuron
        in this layer and store them in self.activation_derivatives.

        """
        if self.activation_function == "relu":
            self.activation_derivatives = self.relu_deriv(self.a_s)
        elif self.activation_function == "sigmoid":
            self.activation_derivatives = self.sigmoid_deriv_deriv(self.a_s)
        else:
            raise ValueError("Unknown activation function")

        return self.activation_derivatives
        

    def update_parameters(self, learning_rate):
        """
        Update the layer's weights and biases using the stored gradients
        and then clear all intermediate variables associated with the previous forward/backward pass.

        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent.
        """
        if self.dweights is None or self.dbiases is None:
            raise ValueError("Gradients not computed. Run compute_gradients first.")

        # update weights and biases 
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

        # cleanup
        self.dweights = None
        self.dbiases = None
        self.delta = None
        self.activation_derivatives = None
        self.z_s = None
        self.a_s = None



def print_weights_and_biases(self):
        print(f"Weights for layer number {self.n}:")
        print(np.round(self.weights, 3))
        print(f"Biases for layer number {self.n}:")
        print(np.round(self.biases, 3))



def test_layer_call():
    """
    Test the __call__ method of ANN_Layer.
    Verifies that the forward pass + activation works correctly.
    """
    # Create a simple layer with 2 inputs and 2 outputs
    layer = ANN_Layer(n=0, n_neurons_input=2, n_neurons_output=2, activation_function="relu")
    
    # Manually set weights and biases for predictable output
    layer.weights = np.array([
        [1, 2],  # neuron 1
        [3, 4]   # neuron 2
    ])
    layer.biases = np.array([
        [1],     # bias for neuron 1
        [-2]     # bias for neuron 2
    ])
    
    # Define a test input vector (column vector)
    input_vector = np.array([
        [1],  # input 1
        [2]   # input 2
    ])
    
    # Call the layer using the __call__ method
    output = layer(input_vector)
    
    # Expected calculation:
    # Neuron 1: max(0, (1*1 + 2*2) + 1) = max(0, 1+4+1) = 6
    # Neuron 2: max(0, (3*1 + 4*2) + (-2)) = max(0, 3+8-2) = 9
    expected_output = np.array([[6], [9]])
    
    print("Output from layer __call__:", output)
    print("Expected output:", expected_output)
    
    assert np.array_equal(output, expected_output), "Test failed: output does not match expected result."
    print("\nTest passed ✅")


# Run the test
if __name__ == "__main__":
    test_layer_call()
    