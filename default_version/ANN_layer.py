##################################
# Build Aritificial Neural Network
##################################
import random

# no seed - each ANN should be initialised with random weights + epochs must differ 

class ANN_Layer():
    
    EULER_NUMBER = 2.718281828459045
    
    ACTIVATION_FUNCTIONS = {
        # *** change from relu to sigmoid
        "relu": {
            "func": lambda x: max(0, x),
            "deriv": lambda x: 1 if x > 0 else 0
        },
        "sigmoid": {
            "func": lambda x: 1 / (1 + ANN_Layer.EULER_NUMBER ** (-x)),
            # deriv can be optimized because digmoid is being computed twice
            "deriv": lambda x: (1 / (1 + ANN_Layer.EULER_NUMBER ** (-x))) *
                           (1 - (1 / (1 + ANN_Layer.EULER_NUMBER ** (-x))))
        }
    }
    
    def __init__(self, n, n_neurons_input, n_neurons_output, activation_function):
        """
        Initialize a layer in the neural network.

        Parameters:
        -----------
        n : int
            Layer index / identifier in the network.
            - Layer 0 is considered the first computational layer:
                it receives the raw input, applies weights, adds biases,
                and then applies the activation function to produce its output.
            - Layer n (>0) similarly receives input from the previous layer
              and produces output after applying weights, biases, and activation.
    
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
        self.weights = []
        self.biases = []
        
        # Intermediate values
        self.z_s = []
        self.a_s = []
        
        # Backpropagation
        self.activation_derivatives = []
        self.delta = []  # to store error signal for backprop
        
    def __mul__(self, input_vector):
        """
        Magic method to allow 'layer * input_vector' syntax.

        Parameters:
        -----------
        input_vector : list of lists
            Input data to the layer (each row is an input vector).

        Returns:
        --------
        list of lists
            Result of multiplying the layer's weight matrix by the input_vector 
            and adding biases (activation can be applied afterward).
        """

        # --- Type checks ---
        if not isinstance(input_vector, list):
            raise TypeError("input_vector must be a list of lists")
        if not all(isinstance(row, list) for row in input_vector):
            raise TypeError("All elements of input_vector must be lists (rows)")

        # --- Value checks ---
        if len(input_vector) == 0:
            raise ValueError("input_vector cannot be empty")
        if any(len(row) != 1 for row in input_vector):
            raise ValueError("Each row in input_vector must contain 1 element")

        # --- Product checks (dimension compatibility) ---
        n_inputs = len(self.weights[0])  # number of inputs expected
        if len(input_vector) != n_inputs:
            raise ValueError(
                f"Input must have exactly {n_inputs} elements "
                "to match the layer's input size."
            )

        # --- Matrix product (weights x input_vector) ---
        output_vector = []

        for row in self.weights:
            output_row = []
            dot_product = sum(row[i] * input_vector[i][0] for i in range(len(row)))
            output_vector.append([dot_product])

        return output_vector    
    
    def __add__(self, output_matrix):
        """
        Magic method to add the layer's biases to a pre-activation output vector.

        Parameters:
        -----------
        output_matrix : list of lists
            Result of the layer multiplication (weights * input), shape (n_neurons_output, 1)
            or multiple inputs (rows = inputs, columns = neurons).

        Returns:
        --------
        list of lists
            The output_matrix with biases added element-wise.
        """

        # --- Type checks ---
        if not isinstance(output_matrix, list) or not all(isinstance(row, list) for row in output_matrix):
            raise TypeError("output_matrix must be a list of lists")

        # --- Value checks ---
        if len(output_matrix) == 0:
            raise ValueError("output_matrix cannot be empty")
        if len(output_matrix) != self.n_neurons_output:
            raise ValueError(
                f"output_matrix must have {self.n_neurons_output} rows to match the biases"
            )
        if any(len(row) != 1 for row in output_matrix):
            raise ValueError("Each row in output_matrix must contain exactly 1 element")

        # --- Add biases ---
        result = [
            [output_matrix[i][0] + self.biases[i][0]]
            for i in range(self.n_neurons_output)
        ]

        return result
    
    def __call__(self, input_vector):
        """
        Enables calling the layer like a function: layer(input_vector).

        Performs the forward pass and returns the activated output.

        Parameters:
        -----------
        input_vector : list of lists
            Input to the layer (column vector or batch of column vectors).

        Returns:
        --------
        list of lists
            Activated output of the layer.
        """
        self.forward(input_vector)           # compute pre-activation z_s
        activated_output = self._apply_activation()  # compute and store a_s
        return activated_output
        
    @property
    def weights_matrix(self):
        """Getter for weights"""
        return self.weights

    @weights_matrix.setter
    def weights_matrix(self, new_weights):
        """Setter for weights"""
        if not isinstance(new_weights, list) or not all(isinstance(row, list) for row in new_weights):
            raise TypeError("Weights must be a list of lists")
        if len(new_weights) != self.n_neurons_output:
            raise ValueError(f"Weights must have {self.n_neurons_output} rows")
        if any(len(row) != self.n_neurons_input for row in new_weights):
            raise ValueError(f"Each weight row must have {self.n_neurons_input} columns")
        self.weights = new_weights

    @property
    def biases_vector(self):
        """Getter for biases"""
        return self.biases

    @biases_vector.setter
    def biases_vector(self, new_biases):
        """Setter for biases"""
        if not isinstance(new_biases, list) or not all(isinstance(row, list) for row in new_biases):
            raise TypeError("Biases must be a list of lists")
        if len(new_biases) != self.n_neurons_output:
            raise ValueError(f"Biases must have {self.n_neurons_output} rows")
        if any(len(row) != 1 for row in new_biases):
            raise ValueError("Each bias row must have exactly 1 column")
        self.biases = new_biases
        
    def _print_matrix(self, matrix):
        """
        Prints a 2D matrix in a neatly formatted way.

        Each element is displayed with 3 decimal places. 
        If an element is None, it prints 'None' instead.

        Parameters:
        -----------
        matrix : list of lists
            - A 2D list representing the matrix to print. Each element should 
            be a number or None.
            - The inside lists are the rows of the matri
        """
        # Iterate through each row of the matrix
        for row in matrix:
            # For each row, join the string representation of each element with a tab
            # Format numbers to 3 decimal places, align them in a width of 8 characters
            # Print 'None' right-aligned if the element is None
            print("\t".join(
                f"{val:8.3f}" if val is not None else f"{'None':>8}"
                for val in row
            ))
        
    def print_weights_and_biases(self):
        """
        Prints the weight and bias matrices for one layer in a neural network.
            
        This is useful for inspecting the parameters of each layer during debugging 
        or analysis. Each weight matrix connects one layer to the next, and each bias 
        matrix corresponds to the neurons of the next layer.
        """
        # Print the weight matrix
        print(f"Weights for layer number {self.n}:")
        self._print_matrix(self.weights)

        # Print the bias matrix
        print(f"Biases for layer number {self.n}:")
        self._print_matrix(self.biases)
            
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
            random.seed(seed)  # Set seed for reproducibility

        # Weight matrix: shape (n_neurons_output, n_neurons_input)
        self.weights = [
            [random.uniform(-0.5, 0.5) for _ in range(self.n_neurons_input)]
            for _ in range(self.n_neurons_output)
        ]

        # Bias vector: shape (n_neurons_output, 1)
        self.biases = [[0.0] for _ in range(self.n_neurons_output)]
    
    def forward(self, input_vector):
        before_bias = self * input_vector  # uses __mul__ for W*x
        after_bias = self + before_bias          # uses __add__ for +b
        self.z_s = after_bias
        return self.z_s
        
    def _apply_activation(self):
        """
        Apply the layer's activation function to each element of self.z_s.
    
        Returns:
        --------
        list of lists
            Activated outputs as column vectors.
        """
        if not hasattr(self, 'z_s'):
            raise ValueError("Pre-activation outputs 'self.z_s' not set")

        func = self.ACTIVATION_FUNCTIONS[self.activation_function]['func']
        self.a_s = [[func(x[0])] for x in self.z_s]
        return self.a_s
        
    def shape(self, what):
        """
        Return the shape of the layer's weights, biases, or output.

        Parameters:
        -----------
        what : str
            What shape to return: "weights", "biases", or "output".
    
        Returns:
        --------
        tuple
            Shape as (rows, columns) for weights/biases, or (n_neurons_output, 1) for output.
        """
        if what == "weights":
            return (len(self.weights), len(self.weights[0]) if self.weights else 0)
        elif what == "biases":
            return (len(self.biases), len(self.biases[0]) if self.biases else 0)
        elif what == "output":
            return (len(self.a_s), len(self.a_s[0]) if self.a_s else 0)
        else:
            raise ValueError("Invalid argument for 'what'. Choose 'weights', 'biases', or 'output'.")

    def compute_activation_derivatives(self):
        """
        Compute the derivative of the activation function for each neuron
        in this layer and store them in self.activation_derivatives.
    
        Returns:
        --------
        list of floats
            Activation derivatives for this layer.
        """
        self.activation_derivatives = []
        deriv_func = self.ACTIVATION_FUNCTIONS[self.activation_function]['deriv']
        for z in self.z_s:
            # z is a single-element list [[value]], so take z[0]
            self.activation_derivatives.append(deriv_func(z[0]))

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
        if not hasattr(self, "dweights") or not hasattr(self, "dbiases"):
            raise ValueError("Gradients not computed. Run compute_gradients first.")

        # --- Update weights ---
        for i in range(self.n_neurons_output):
            for j in range(self.n_neurons_input):
                self.weights[i][j] -= learning_rate * self.dweights[i][j]

        # --- Update biases ---
        for i in range(self.n_neurons_output):
            self.biases[i][0] -= learning_rate * self.dbiases[i]

        # --- Clean up temporary variables ---
        self.dweights = None
        self.dbiases = None
        self.delta = None
        self.activation_derivatives = None
        self.z_s = None
        self.a_s = None

