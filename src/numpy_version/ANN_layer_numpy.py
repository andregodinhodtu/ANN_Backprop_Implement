import numpy as np
import random


class ANN_Layer_numpy():

    ACTIVATION_FUNCTIONS = {
        "relu": {
            "func": lambda x: np.maximum(0, x),
            "deriv": lambda x: (x > 0).astype(float)
        },
        "sigmoid": {
            "func": lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            "deriv": lambda x: (lambda s: s * (1 - s))(
                1 / (1 + np.exp(-np.clip(x, -500, 500)))
            )
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
        n : int
            Layer index.
        n_neurons_input : int
            Number of input neurons coming into this layer
        n_neurons_output : int
            Number of neurons in this layer (after activation).
        activation_function : str
            Activation function to apply to this layer's output.
            Must be a key in `ACTIVATION_FUNCTIONS`.
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

        #  Intermediate values (forward pass)
        self.z_s = None
        self.a_s = None

        # --- Backpropagation ---
        self.activation_derivatives = None
        # error signal for backprop
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
        if not isinstance(new_weights, (list, np.ndarray)):
            raise TypeError("Weights must be a list of lists or a numpy array")
        if isinstance(new_weights, list) and not all(isinstance(row, list) for row in new_weights):
            raise TypeError("All elements of weights must be lists (rows)")
        if isinstance(new_weights, list) and not all(
            isinstance(val, (int, float)) for row in new_weights for val in row
        ):
            raise TypeError("All values in weights must be ints or floats")
        if isinstance(new_weights, np.ndarray) and not np.issubdtype(new_weights.dtype, np.number):
            raise TypeError("All values in weights must be numeric")

        new_weights = np.array(new_weights)

        if new_weights.ndim != 2:
            raise ValueError("Weights must be a 2D matrix")
        if new_weights.shape != (self.n_neurons_output, self.n_neurons_input):
            raise ValueError(
                f"Weights must have shape ({self.n_neurons_output}, {self.n_neurons_input})"
            )

        self._weights = new_weights

    @property
    def biases(self):
        """Getter for biases."""
        return self._biases

    @biases.setter
    def biases(self, new_biases):
        """Validate and set the bias vector."""
        if not isinstance(new_biases, (list, np.ndarray)):
            raise TypeError("Biases must be a list of lists or a numpy array")
        if isinstance(new_biases, list) and not all(isinstance(row, list) for row in new_biases):
            raise TypeError("All elements of biases must be lists (rows)")
        if isinstance(new_biases, list) and not all(
            isinstance(val, (int, float)) for row in new_biases for val in row
        ):
            raise TypeError("All values in biases must be ints or floats")
        if isinstance(new_biases, np.ndarray) and not np.issubdtype(new_biases.dtype, np.number):
            raise TypeError("All values in biases must be numeric")

        new_biases = np.array(new_biases)

        if new_biases.ndim != 2:
            raise ValueError("Biases must be a 2D column vector")
        if new_biases.shape != (self.n_neurons_output, 1):
            raise ValueError(f"Biases must have shape ({self.n_neurons_output}, 1)")

        self._biases = new_biases

    def print_weights_and_biases(self):
        """Pretty-print the layer's weights and biases."""
        print(f"Weights for layer number {self.n}:")
        print(np.round(self.weights, 3))
        print(f"Biases for layer number {self.n}:")
        print(np.round(self.biases, 3))

    def initialize_weights_bias(self, rng=None):
        """
        Initialize weights based on the activation function:
        - ReLU / Leaky ReLU  → He initialization
        - Sigmoid / Tanh     → Xavier / Glorot initialization
        Biases are initialized to 0.

        Parameters
        ----------
        rng : np.random.Generator or None
            Optional NumPy random generator for reproducibility.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Pick initialization strategy based on activation function
        if self.activation_function in ("relu", "leaky_relu"):
            # He initialization
            std = np.sqrt(2 / self.n_neurons_input)
        else:
            # Xavier / Glorot initialization (sigmoid, tanh, etc.)
            std = np.sqrt(2 / (self.n_neurons_input + self.n_neurons_output))

        # Assignments go through the setters (shape/type validation runs)
        self.weights = rng.normal(
            loc=0.0,
            scale=std,
            size=(self.n_neurons_output, self.n_neurons_input),
        )
        self.biases = np.zeros((self.n_neurons_output, 1))

    def forward(self, input_vector):
        """
        Compute the full forward pass of the layer: a = f(W @ x + b).

        Parameters:
        input_vector : np.ndarray or list of lists
            Column vector of shape (n_neurons_input, 1) 
            or 
            batch (n_neurons_input, batch_size).

        Returns:
        np.ndarray
            Activated output a, shape (n_neurons_output, 1) 
            or
            (n_neurons_output, batch_size).
        """
        # Type checks
        if not isinstance(input_vector, (list, np.ndarray)):
            raise TypeError("input_vector must be a list of lists or a numpy array")
        if isinstance(input_vector, list) and not all(isinstance(row, list) for row in input_vector):
            raise TypeError("All elements of input_vector must be lists (rows)")
        if isinstance(input_vector, list) and not all(
            isinstance(val, (int, float)) for row in input_vector for val in row
        ):
            raise TypeError("All values in input_vector must be ints or floats")
        if isinstance(input_vector, np.ndarray) and not np.issubdtype(input_vector.dtype, np.number):
            raise TypeError("All values in input_vector must be numeric")

        input_vector = np.array(input_vector)

        # Value checks
        if input_vector.size == 0:
            raise ValueError("input_vector cannot be empty")
        if input_vector.ndim != 2:
            raise ValueError(
                f"input_vector must be 2D with shape (n_in, batch_size). "
                f"Got {input_vector.ndim}D array with shape {input_vector.shape}."
            )
        if input_vector.shape[0] != self.n_neurons_input:
            raise ValueError(
                f"input_vector must have {self.n_neurons_input} rows "
                f"(one per input neuron), got {input_vector.shape[0]}."
            )
        if input_vector.shape[1] == 0:
            raise ValueError("Batch size cannot be 0.")

        # State checks
        if self._weights is None:
            raise ValueError("Weights are not initialized. Run initialize_weights_bias() first.")
        if self._biases is None:
            raise ValueError("Biases are not initialized. Run initialize_weights_bias() first.")

        # z = W @ x + b   (biases broadcast across batch)
        self.z_s = self._weights @ input_vector + self._biases

        # a = f(z)
        func = self.ACTIVATION_FUNCTIONS[self.activation_function]['func']
        self.a_s = func(self.z_s)
        return self.a_s

    def compute_activation_derivatives(self):
        """
        Compute and store f'(z) for this layer.
        """
        if self.z_s is None:
            raise ValueError("z_s is not computed. Run forward() first.")
        if len(self.z_s) == 0:
            raise ValueError("z_s is empty.")
        
        # Fetch the activation function this layer was initialized with
        deriv_func = self.ACTIVATION_FUNCTIONS[self.activation_function]['deriv']
        self.activation_derivatives = deriv_func(self.z_s)
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

        # In-place updates on the backing fields.
        # Shape invariant is preserved by construction (same-shape arithmetic),
        # Skip the setter to avoid redundant validation 
        self._weights -= learning_rate * (self.dweights + l2_lambda * self._weights)
        self._biases -= learning_rate * self.dbiases

        # Clean up temporary variables
        self.dweights = None
        self.dbiases = None
        self.delta = None
        self.activation_derivatives = None
        self.z_s = None
        self.a_s = None