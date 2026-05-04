import random
import math


class ANN_Layer_base_python():
    
    ACTIVATION_FUNCTIONS = {
        "relu": {
            "func": lambda x: max(0, x),
            "deriv": lambda x: 1 if x > 0 else 0
        },
        "sigmoid": {
            "func":  lambda x: 1 / (1 + math.exp(-max(min(x, 500), -500))),
            "deriv": lambda x: (lambda s: s * (1 - s))(
                1 / (1 + math.exp(-max(min(x, 500), -500)))
            ),
        },
        "leaky_relu": {
            "func": lambda x: x if x > 0 else 0.01 * x,
            "deriv": lambda x: 1 if x > 0 else 0.01
        }
    }

    def __init__(self, n, n_neurons_input, n_neurons_output, activation_function):
        """
        Initialize a layer in the neural network.

        Parameters:
        n : int
            Layer index 
        n_neurons_input : int
            Number of input neurons coming into this layer.
        n_neurons_output : int
            Number of neurons in this layer (after activation).
        activation_function : str
            Activation function. Must be a key in `ACTIVATION_FUNCTIONS`.
        """

        # Type checks
        if not isinstance(n, int):
            raise TypeError("n must be an Integer")
        if not isinstance(n_neurons_input, int):
            raise TypeError("n_neurons_input must be an Integer")
        if not isinstance(n_neurons_output, int):
            raise TypeError("n_neurons_output must be an Integer")
        if not isinstance(activation_function, str):
            raise TypeError("activation_function should be a String")

        # Value checks
        if n < 0:
            raise ValueError("n must be >= 0")
        if n_neurons_input <= 0:
            raise ValueError("n_neurons_input must be > 0")
        if n_neurons_output <= 0:
            raise ValueError("n_neurons_output must be > 0")
        if activation_function not in self.ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unknown activation function: {activation_function}")

        # Layer structure
        self.n = n
        self.n_neurons_input = n_neurons_input
        self.n_neurons_output = n_neurons_output
        self.activation_function = activation_function

        # Parameters (backing storage for the properties)
        self._weights = None
        self._biases = None

        # Intermediate values (forward pass)
        self.z_s = None
        self.a_s = None

        # Backpropagation
        self.activation_derivatives = None
        self.delta = None
        self.dweights = None
        self.dbiases = None

    def __call__(self, input_vector):
        """
        Enables calling the layer like a function: layer(input_vector).
        Performs the forward pass and returns the activated output.
        """
        # useful outside, code intution in ANN
        return self.forward(input_vector)

    @property
    def weights(self):
        """Getter for weights."""
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        """Validate and set the weight matrix."""
        if not isinstance(new_weights, list) or not all(isinstance(row, list) for row in new_weights):
            raise TypeError("Weights must be a list of lists")
        if not all(isinstance(val, (int, float)) for row in new_weights for val in row):
            raise TypeError("All values in weights must be ints or floats")
        if len(new_weights) != self.n_neurons_output:
            raise ValueError(f"Weights must have {self.n_neurons_output} rows")
        if any(len(row) != self.n_neurons_input for row in new_weights):
            raise ValueError(f"Each weight row must have {self.n_neurons_input} columns")

        self._weights = new_weights

    @property
    def biases(self):
        """Getter for biases."""
        return self._biases

    @biases.setter
    def biases(self, new_biases):
        """Validate and set the bias vector."""
        if not isinstance(new_biases, list) or not all(isinstance(row, list) for row in new_biases):
            raise TypeError("Biases must be a list of lists")
        if not all(isinstance(val, (int, float)) for row in new_biases for val in row):
            raise TypeError("All values in biases must be ints or floats")
        if len(new_biases) != self.n_neurons_output:
            raise ValueError(f"Biases must have {self.n_neurons_output} rows")
        if any(len(row) != 1 for row in new_biases):
            raise ValueError("Each bias row must have exactly 1 column")

        self._biases = new_biases

    def _print_matrix(self, matrix):
        """Pretty-print a 2D matrix with 3 decimal places per cell."""
        for row in matrix:
            print("\t".join(
                f"{val:8.3f}" if val is not None else f"{'None':>8}"
                for val in row
            ))

    def print_weights_and_biases(self):
        """Pretty-print the layer's weights and biases."""
        print(f"Weights for layer number {self.n}:")
        self._print_matrix(self.weights)
        print(f"Biases for layer number {self.n}:")
        self._print_matrix(self.biases)

    def _shape(self, what):
        """Return the shape of the layer's weights, biases, or output."""
        if what == "weights":
            return (len(self._weights), len(self._weights[0]) if self._weights else 0)
        elif what == "biases":
            return (len(self._biases), len(self._biases[0]) if self._biases else 0)
        elif what == "output":
            return (len(self.a_s), len(self.a_s[0]) if self.a_s else 0)
        else:
            raise ValueError("Invalid argument for 'what'. Choose 'weights', 'biases', or 'output'.")

    def initialize_weights_bias(self, rng=None):
        """
        Initialize weights based on the activation function:
        - ReLU / Leaky ReLU  → He initialization
        - Sigmoid / Tanh     → Xavier / Glorot initialization
        Biases are initialized to 0.

        Parameters
        ----------
        rng : random.Random or None
            Random number generator instance. If None, a new local Random()
            is created (non-deterministic).
        """
        if rng is None:
            rng = random.Random()
        
        # Pick initialization strategy based on activation function
        if self.activation_function in ("relu", "leaky_relu"):
            # He initialization
            std = math.sqrt(2 / self.n_neurons_input)
        else:c
            # Xavier / Glorot initialization
            std = math.sqrt(2 / (self.n_neurons_input + self.n_neurons_output))

        # Assignments go through the setters (shape/type validation runs)
        self.weights = [
            [rng.gauss(0, std) for _ in range(self.n_neurons_input)]
            for _ in range(self.n_neurons_output)
        ]
        self.biases = [[0.0] for _ in range(self.n_neurons_output)]

    def _matrix_multiply(self, input_vector):
        """Multiply the weight matrix by the input vector (column form)."""
        output_vector = []
        for row in self._weights:
            dot_product = sum(row[i] * input_vector[i][0] for i in range(len(row)))
            output_vector.append([dot_product])
        return output_vector

    def _add_biases(self, output_matrix):
        """Add the bias vector after matrix multiply."""
        return [
            [output_matrix[i][0] + self._biases[i][0]]
            for i in range(self.n_neurons_output)
        ]

    def forward(self, input_vector):
        """
        Compute the full forward pass of the layer: a = f(W * x + b).

        Parameters:
        input_vector : list of lists
            Column vector of shape (n_neurons_input, 1).

        Returns:
        list of lists
            Activated output a, shape (n_neurons_output, 1).
        """
        # Type checks
        if not isinstance(input_vector, list):
            raise TypeError("input_vector must be a list of lists")
        if not all(isinstance(row, list) for row in input_vector):
            raise TypeError("All elements of input_vector must be lists (rows)")
        if not all(isinstance(val, (int, float)) for row in input_vector for val in row):
            raise TypeError("All values in input_vector must be ints or floats")

        # Value checks
        if len(input_vector) == 0:
            raise ValueError("input_vector cannot be empty")
        if any(len(row) != 1 for row in input_vector):
            raise ValueError("Each row in input_vector must contain 1 element")

        # State checks
        if self._weights is None:
            raise ValueError("Weights are not initialized. Run initialize_weights_bias() first.")
        if self._biases is None:
            raise ValueError("Biases are not initialized. Run initialize_weights_bias() first.")

        # Dimension compatibility check
        n_inputs = len(self._weights[0])
        if len(input_vector) != n_inputs:
            raise ValueError(
                f"Input must have exactly {n_inputs} elements "
                "to match the layer's input size."
            )

        # z = W * x + b
        before_bias = self._matrix_multiply(input_vector)
        self.z_s = self._add_biases(before_bias)

        # a = f(z)
        func = self.ACTIVATION_FUNCTIONS[self.activation_function]['func']
        self.a_s = [[func(x[0])] for x in self.z_s]

        return self.a_s

    def compute_activation_derivatives(self):
        """Compute and store f'(z) for this layer."""
        if self.z_s is None:
            raise ValueError("z_s is not computed. Run forward() first.")
        if len(self.z_s) == 0:
            raise ValueError("z_s is empty.")
        
        # Fetch the activation function this layer was initialized with
        deriv_func = self.ACTIVATION_FUNCTIONS[self.activation_function]['deriv']
        self.activation_derivatives = [deriv_func(z[0]) for z in self.z_s]
        return self.activation_derivatives

    def update_parameters(self, learning_rate, l2_lambda=0.0):
        """
        Update weights and biases using the stored gradients with optional
        L2 regularization, then clear intermediate variables.

        Parameters:
        
        learning_rate : float
            Learning rate for gradient descent.
        l2_lambda : float
            L2 regularization coefficient. Default 0.0 (no regularization).
        """
        # Type checks 
        if not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be a number")
        if not isinstance(l2_lambda, (int, float)):
            raise TypeError("l2_lambda must be a number")

        # Value checks
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if l2_lambda < 0:
            raise ValueError("l2_lambda must be >= 0")

        # State checks
        if self.dweights is None:
            raise ValueError("Gradients not computed. Run compute_gradients first.")
        if self.dbiases is None:
            raise ValueError("Gradients not computed. Run compute_gradients first.")

        # Update weights with L2 regularization.
        # Skip the setter to avoid redundant validation in this hot loop
        self._weights = [
            [
                self._weights[i][j] - learning_rate * (self.dweights[i][j] + l2_lambda * self._weights[i][j])
                for j in range(self.n_neurons_input)
            ]
            for i in range(self.n_neurons_output)
        ]
        
        # Update biases (no regularization).
        self._biases = [
            [self._biases[i][0] - learning_rate * self.dbiases[i][0]]
            for i in range(self.n_neurons_output)
        ]

        # Clean up temporary variables
        self.dweights = None
        self.dbiases = None
        self.delta = None
        self.activation_derivatives = None
        self.z_s = None
        self.a_s = None
