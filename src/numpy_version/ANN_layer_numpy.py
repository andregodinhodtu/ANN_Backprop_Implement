import numpy as np
import random

# no seed - each ANN should be initialised with random weights + epochs must differ 

class ANN_Layer_numpy():
    
    ACTIVATION_FUNCTIONS = {
        "relu": {
            "func": lambda x: np.maximum(0, x),
            "deriv": lambda x: (x > 0).astype(float)
        },
        "sigmoid": {
            "func": lambda x: 1 / (1 + np.e ** (-x)),
            "deriv": lambda x: (1 / (1 + np.e ** (-x))) *
                               (1 - (1 / (1 + np.e ** (-x))))
        },
        "leaky_relu": {
            "func": lambda x: np.where(x > 0, x, 0.01 * x),
            "deriv": lambda x: np.where(x > 0, 1.0, 0.01)
        }
    }
    
    def __init__(self, n, n_neurons_input, n_neurons_output, activation_function):
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
            Must be in `self.ACTIVATION_FUNCTIONS`.
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
        if activation_function not in self.ACTIVATION_FUNCTIONS:
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
        # compute pre-activation z_s + activation a_s
        return self.forward(input_vector)       
        
    @property
    def weights_matrix(self):
        """Getter for weights"""
        return self.weights

    @weights_matrix.setter
    def weights_matrix(self, new_weights):
        """Setter for weights"""
        if not isinstance(new_weights, (list, np.ndarray)):
            raise TypeError("Weights must be a list of lists or a numpy array")
        if isinstance(new_weights, list) and not all(isinstance(row, list) for row in new_weights):
            raise TypeError("All elements of weights must be lists (rows)")
        if isinstance(new_weights, list) and not all(isinstance(val, (int, float)) for row in new_weights for val in row):
            raise TypeError("All values in weights must be ints or floats")
        if isinstance(new_weights, np.ndarray) and not np.issubdtype(new_weights.dtype, np.number):
            raise TypeError("All values in weights must be numeric")
        new_weights = np.array(new_weights)
        if new_weights.ndim != 2:
            raise ValueError("Weights must be a 2D matrix")
        if new_weights.shape != (self.n_neurons_output, self.n_neurons_input):
            raise ValueError(f"Weights must have shape ({self.n_neurons_output}, {self.n_neurons_input})")
        self.weights = new_weights

    @property
    def biases_vector(self):
        """Getter for biases"""
        return self.biases

    @biases_vector.setter
    def biases_vector(self, new_biases):
        """Setter for biases"""
        if not isinstance(new_biases, (list, np.ndarray)):
            raise TypeError("Biases must be a list of lists or a numpy array")
        if isinstance(new_biases, list) and not all(isinstance(row, list) for row in new_biases):
            raise TypeError("All elements of biases must be lists (rows)")
        if isinstance(new_biases, list) and not all(isinstance(val, (int, float)) for row in new_biases for val in row):
            raise TypeError("All values in biases must be ints or floats")
        if isinstance(new_biases, np.ndarray) and not np.issubdtype(new_biases.dtype, np.number):
            raise TypeError("All values in biases must be numeric")
        new_biases = np.array(new_biases)
        if new_biases.ndim != 2:
            raise ValueError("Biases must be a 2D column vector")
        if new_biases.shape != (self.n_neurons_output, 1):
            raise ValueError(f"Biases must have shape ({self.n_neurons_output}, 1)")
        self.biases = new_biases
     
    def print_weights_and_biases(self):
        """
        Prints the weight and bias matrices for one layer in a neural network.
            
        This is useful for inspecting the parameters of each layer during debugging 
        or analysis. Each weight matrix connects one layer to the next, and each bias 
        matrix corresponds to the neurons of the next layer.
        """
        print(f"Weights for layer number {self.n}:")
        print(np.round(self.weights, 3))
        print(f"Biases for layer number {self.n}:")
        print(np.round(self.biases, 3)) 
                   
    def initialize_weights_bias(self, seed=None):
        """
        Initialize weights based on the activation function:
        - ReLU / Leaky ReLU → He initialization
        - Sigmoid / Tanh     → Xavier / Glorot initialization
        Biases are initialized to 0.

        Parameters:
        -----------
        seed : int or None
            Optional seed for the random number generator to make results reproducible.
        """
        rng = np.random.default_rng(seed)

        # Pick initialization strategy based on activation function
        if self.activation_function in ("relu", "leaky_relu"):
            # He initialization
            std = np.sqrt(2 / self.n_neurons_input)
        else:
            # Xavier / Glorot initialization (sigmoid, tanh, etc.)
            std = np.sqrt(2 / (self.n_neurons_input + self.n_neurons_output))

        # Weight matrix: shape (n_neurons_output, n_neurons_input)
        self.weights = rng.normal(0, std, size=(self.n_neurons_output, self.n_neurons_input))

        # Bias vector: shape (n_neurons_output, 1)
        self.biases = np.zeros((self.n_neurons_output, 1))

    def forward(self, input_vector):
        """
        Compute the full forward pass of the layer: a = f(W * x + b).
        Parameters:
        -----------
        input_vector : np.ndarray or list of lists
            Column vector of shape (n_neurons_input, 1).
        Returns:
        --------
        np.ndarray
            Activated output a, shape (n_neurons_output, 1).
        """
        # --- Type checks ---
        if not isinstance(input_vector, (list, np.ndarray)):
            raise TypeError("input_vector must be a list of lists or a numpy array")
        if isinstance(input_vector, list) and not all(isinstance(row, list) for row in input_vector):
            raise TypeError("All elements of input_vector must be lists (rows)")
        if isinstance(input_vector, list) and not all(isinstance(val, (int, float)) for row in input_vector for val in row):
            raise TypeError("All values in input_vector must be ints or floats")
        if isinstance(input_vector, np.ndarray) and not np.issubdtype(input_vector.dtype, np.number):
            raise TypeError("All values in input_vector must be numeric")
            
        # Convert to numpy if needed
        input_vector = np.array(input_vector)

        # --- Value checks ---
        if input_vector.ndim != 2:
            raise ValueError("input_vector must be a 2D array")
        if input_vector.shape[1] != 1:
            raise ValueError("input_vector must have exactly 1 column")

        # --- State checks ---
        if self.weights is None:
            raise ValueError("Weights are not initialized. Run initialize_weights_bias() first.")
        if self.biases is None:
            raise ValueError("Biases are not initialized. Run initialize_weights_bias() first.")

        # --- Dimension compatibility check ---
        if input_vector.shape[0] != self.n_neurons_input:
            raise ValueError(
                f"Input must have exactly {self.n_neurons_input} elements "
                "to match the layer's input size."
            )

        # Compute z = W * x + b and store for backpropagation
        self.z_s = self.weights @ input_vector + self.biases

        # Compute a = f(z) and store for backpropagation
        func = self.ACTIVATION_FUNCTIONS[self.activation_function]['func']
        self.a_s = func(self.z_s)
        return self.a_s

    def compute_activation_derivatives(self):
        """
        Compute the derivative of the activation function for each neuron
        in this layer and store them in self.activation_derivatives.
        Returns:
        --------
        np.ndarray
            Activation derivatives for this layer.
        """
        # --- State checks ---
        if self.z_s is None:
            raise ValueError("z_s is not computed. Run forward() first.")

        deriv_func = self.ACTIVATION_FUNCTIONS[self.activation_function]['deriv']
        self.activation_derivatives = deriv_func(self.z_s)
        return self.activation_derivatives
        
    def update_parameters(self, learning_rate, l2_lambda=0.0):
        """
        Update the layer's weights and biases using the stored gradients
        with optional L2 regularization, then clear intermediate variables.
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent.
        l2_lambda : float
            L2 regularization coefficient (weight decay). Default 0.0 (no regularization).
        """
        # --- Type checks ---
        if not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be a number")
        if not isinstance(l2_lambda, (int, float)):
            raise TypeError("l2_lambda must be a number")

        # --- Value checks ---
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if l2_lambda < 0:
            raise ValueError("l2_lambda must be >= 0")

        # --- State checks ---
        if not hasattr(self, "dweights") or self.dweights is None:
            raise ValueError("Gradients not computed. Run compute_gradients first.")
        if not hasattr(self, "dbiases") or self.dbiases is None:
            raise ValueError("Gradients not computed. Run compute_gradients first.")

        # --- Update weights with L2 regularization ---
        self.weights -= learning_rate * (self.dweights + l2_lambda * self.weights)

        # --- Update biases (no regularization) ---
        self.biases -= learning_rate * self.dbiases

        # --- Clean up temporary variables ---
        self.dweights = None
        self.dbiases = None
        self.delta = None
        self.activation_derivatives = None
        self.z_s = None
        self.a_s = None
    