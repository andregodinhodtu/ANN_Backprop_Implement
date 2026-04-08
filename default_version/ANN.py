##################################
# Build Aritificial Neural Network
##################################
import random
from ANN_layer import ANN_Layer

class ANN():
    
    EULER_NUMBER = 2.718281828459045
    
    ACTIVATION_FUNCTIONS = {
        "relu": {
            "func": lambda x: max(0, x),
            "deriv": lambda x: 1 if x > 0 else 0
        },
        "sigmoid": {
            "func": lambda x: 1 / (1 + ANN.EULER_NUMBER ** (-x)),
            # deriv can be optimized because digmoid is being computed twice
            "deriv": lambda x: (1 / (1 + ANN.EULER_NUMBER ** (-x))) *
                           (1 - (1 / (1 + ANN.EULER_NUMBER ** (-x))))
        }
    
    }
    
    LOSS_FUNCTIONS = {
        "MSE": {
            "func": lambda x, y: sum(
                (x[i][0] - y[i][0]) ** 2 for i in range(len(x))
            ),
            "deriv": lambda x, y: 2 * (x - y)
        },
        "BinaryCrossEntropy" : {
            "func": lambda x, y: sum( - (y[i][0] * ANN.log(max(x[i][0], 1e-15)) +
                                    (1 - y[i][0]) * ANN.log(max(1 - x[i][0], 1e-15)))
                                    for i in range(len(x))) / len(x),  # divide by N
            "deriv": lambda x, y: (x - y) / (x * (1 - x) + 1e-15)
        }
    }
    
    @classmethod
    def log(cls, x, iterations=100):
        """Compute natural log using Newton-Raphson method for ln(x)"""
        if x <= 0:
            raise ValueError("log undefined for non-positive numbers")
        
        y = x - 1.0  # initial guess
        for _ in range(iterations):
            y = y - (cls.EULER_NUMBER**y - x) / (cls.EULER_NUMBER**y)
        return y
        
    
    def __init__(self, n_layers, n_neurons_each_layer, activation_hidden="relu",
                 activation_output="sigmoid", loss_function="MSE"):
        """
        Build a feedforward neural network with n_layers.
    
        Parameters:
        -----------
        n_layers : int
            Number of layers (including output).
        n_neurons_each_layer : list of ints
            Number of neurons in each layer. Length must equal n_layers.
        activation_hidden : str
            Activation function for hidden layers.
        activation_output : str
            Activation function for the output layer.
        loss_function : str
            Loss function to use.
        """
        # Input validation
        if n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if len(n_neurons_each_layer) != n_layers:
            raise ValueError("Length of n_neurons_each_layer must equal n_layers")
    
        # Store activation functions and loss
        self.n_layers = n_layers
        self.n_neurons_each_layer = n_neurons_each_layer
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.loss_function = loss_function
        
        # Layers container
        self.layers = []
    
        # Build layers
        self._build_ANN()
        
    def _build_ANN(self):
        """Private method to construct the layers of the network."""
        for i in range(self.n_layers -1):
            # Number of inputs for this layer
            n_input = self.n_neurons_each_layer[i]
            # Number of neurons in this layer
            n_output = self.n_neurons_each_layer[i+1]
            # Choose activation
            act = self.activation_hidden if i == self.n_layers-2 else self.activation_output
            
            # Create ANN_Layer
            layer = ANN_Layer(
                n=i,
                n_neurons_input=n_input,
                n_neurons_output=n_output,
                activation_function=act
            )
            
            # Initialize weights and biases (optional fixed seed)
            layer.initialize_weights_bias(seed=42)
            
            # Add to layers list
            self.layers.append(layer)
            
    def prediction(self, input_vector):
        """
        Make a forward pass through the entire ANN.

        Parameters:
        -----------
        input_vector : list of lists
            Input column vector (shape: n_input x 1).

        Returns:
        --------
        list of lists
            Output of the last layer after activation.
        """
        working_vector = input_vector

        # Forward pass through all layers
        for layer in self.layers:
            # Use the layer's __call__ to do forward pass and activation
            working_vector = layer(working_vector)

        return working_vector
        
    def _compute_deltas(self, y):
        """
        Compute delta values for each layer in the network for backpropagation.
        Stores them in each layer's `.delta` attribute.

        Parameters:
        -----------
        y : list of lists
            Target output column vector (shape: n_output x 1)
        """
        # Start from output layer and go backward
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            if i == len(self.layers) - 1:
                # Output layer
                layer.delta = []
                for j in range(layer.n_neurons_output):
                    a = layer.a_s[j][0]           # activation
                    target = y[j][0]              # true value
                    dz = self.LOSS_FUNCTIONS[self.loss_function]["deriv"](a, target) * layer.ACTIVATION_FUNCTIONS[layer.activation_function]['deriv'](layer.z_s[j][0])
                    layer.delta.append(dz)
            else:
                # Hidden layers
                next_layer = self.layers[i + 1]
                layer.delta = []
                for j in range(layer.n_neurons_output):
                    # Sum over next layer's deltas weighted by corresponding weights
                    weighted_sum = sum(
                        next_layer.delta[k] * next_layer.weights[k][j]
                        for k in range(next_layer.n_neurons_output)
                    )
                    dz = weighted_sum * layer.ACTIVATION_FUNCTIONS[layer.activation_function]['deriv'](layer.z_s[j][0])
                    layer.delta.append(dz)

    def _build_backprop_matrix(self):
        """ 
        Creates List structure that stores the derivatives of the Loss in function
        of the weights and the biasas 
        """

        # make sure the lists are empty
        self.loss_deriv_of_weights = []
        self.loss_deriv_of_bias = [] 
        
        # Build the structure of the ANN    
        for i in range(self.n_layers - 1):
            
            weight_matrix = [[None for x in range(self.n_neurons_each_layer[i])] for j in range(self.n_neurons_each_layer[i+1])]
            self.loss_deriv_of_weights.append(weight_matrix)
           
            bias_vector = [[None] for x in range(self.n_neurons_each_layer[i+1])]
            self.loss_deriv_of_bias.append(bias_vector)

            """print(f"Weight matrix of Loss derivative between layer {i} and layer {i+1}:")
            self._print_matrix(weight_matrix)
            print(f"Bias Matrix of Loss derivative between layer {i} and layer {i+1}:")
            self._print_matrix(bias_vector)"""
            
    def _fill_with_none(self, matrix):
        """Modify the given matrix in-place, replacing all values with None"""
    
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = None
    
    
    def _backpropagation(self, y):
        
        # for eah layer i want to run trough all the values of parameters in that layer
        #that means running trough each next layer neurosn and inside for each layer neuron
        
        # First we will compute all the delta
        
        working_y = y
        self.delta_s = [[] for _ in range(self.n_layers)]
    
        for layer in range(self.n_layers-1,-1,-1):

            if layer == self.n_layers-1:
                for neuron in range(self.n_neurons_each_layer[layer]):
                    
                    # Gradient of the loss with respect to the output layer activations
                    # dL/da (last activation)
                    a = self.a_s[layer][neuron][0]
                    y = working_y[neuron][0]
                    deriv_loss = self.loss_function_deriv(a,y)
                    
                    # Derivative of the activation function with respect to z (pre-activation)
                    # da/dz
                    z = self.z_s[layer][neuron][0]
                    deriv_activation = self.activation_output_deriv(z)
                    
                    delta = deriv_loss * deriv_activation
                    self.delta_s[layer].append(delta)
                    a = self.a_s[layer][neuron][0]
                    y = working_y[neuron][0]
                    

                    # this is ok only for the output layer, generalization
                    # Simplified gradient for sigmoid + BCE
                   # delta = a - y
                   # self.delta_s[layer].append(delta)
                    
            else:
                   # skip layer 0 — it's the input, no activation to differentiate
                if layer == 0:
                    continue

                for neuron in range(self.n_neurons_each_layer[layer]):
                    sum_over_next_layer_neurons = 0
                    for next_layer_neuron in range(self.n_neurons_each_layer[layer+1]):
            
                        # Activation of the next layer
                        delta_next_layer = self.delta_s[layer+1][next_layer_neuron]
                        
                        # Weight of the explicit transition between next neuron and current neuron
                        weight = self.weights[layer][next_layer_neuron][neuron]
                        
                        sum_over_next_layer_neurons += delta_next_layer * weight
                    
                    # Derivative of the activation function with respect to z (pre-activation)
                    # da/dz
                   # print(f"layer={layer}, len(z_s)={len(self.z_s)}, neuron={neuron}")

                    z = self.z_s[layer][neuron][0]
                    deriv_activation = self.activation_hidden_deriv(z)
                     
                    delta = sum_over_next_layer_neurons * deriv_activation
                    self.delta_s[layer].append(delta)
                    
        # Now let´s fill in the  self.loss_deriv_of_weights and self.loss_deriv_of_bias
        for layer in range(self.n_layers-1):
            for neuron in range(self.n_neurons_each_layer[layer]):
                for next_layer_neuron in range(self.n_neurons_each_layer[layer+1]): 
                    self.loss_deriv_of_weights[layer][next_layer_neuron][neuron] = self.a_s[layer][neuron][0] * self.delta_s[layer+1][next_layer_neuron]
                    self.loss_deriv_of_bias[layer][next_layer_neuron][0] = self.delta_s[layer+1][next_layer_neuron]
                    
        
    def backprop_one_training_example(self, input_vector, y, step = 1000, learning_rate = 0.005, verbose=False):
        """Apply Chain Rule
           We want to calculate the derivatives for the Lost in
           function of the Weights and the biases"""
        
        working_y = self._tranpose_matrix(y)
        
        # print(f"\n" + "#"*25 + " Init Backprop Matrixes " + "#"*25 + "\n")
        self._build_backprop_matrix()
        
        network_response = self.prediction(input_vector)
        
        if verbose:
            print(f"\n" + "#"*28 + " 1º Network Response " + "#"*28 + "\n")
            self._print_matrix(network_response)
            print(f"Error for this cycle is {self.loss_function(working_y, network_response):.4f} \n")
            
        for i in range(step):
            if verbose:
                print(f"-"*20 + f" Backpropagation cycle number {i} " + "-"*20)
            
            # Start by performing backpropagation in the current weights and activations
            self._backpropagation(working_y)

            if verbose:
                print(f"\n" + "#"*28 + " 1º Cycle Backprop " + "#"*28)
                print(f"#"*7 + " Values straight out of backprop, before any product to lr  " + "#"*7 + "\n")
                self._print_weight_and_bias_matrices(self.loss_deriv_of_weights,self.loss_deriv_of_bias)
            
            # Change weights 
        
            adjusted_weight = []
            adjusted_bias = []
        
            for matrix in self.loss_deriv_of_weights:
                adjusted_weight.append(self._product_by_scalar(learning_rate, matrix))
            
            for bias in self.loss_deriv_of_bias:
                adjusted_bias.append(self._product_by_scalar(learning_rate, bias))
            
            
            result_weight_matrix = []
            result_bias_matrix = []
            for j in range(self.n_layers -1):
                result_weight_matrix.append(self._subtract_matrices(self.weights[j],adjusted_weight[j]))
                result_bias_matrix.append(self._subtract_matrices(self.bias[j],adjusted_bias[j]))
            
            if verbose:
                print(f"\n" + "#"*30 + " New parameters " + "#"*30 + "\n")
                self._print_weight_and_bias_matrices(result_weight_matrix,result_bias_matrix)
            
            self.weights = self._copy_structure(result_weight_matrix)
            self.bias = self._copy_structure(result_bias_matrix)
            

            # Reset matrixes
            for matrix in self.loss_deriv_of_weights:
                self._fill_with_none(matrix)

            for matrix in self.loss_deriv_of_bias:
                self._fill_with_none(matrix)
                
            
            network_response = self.prediction(input_vector)
            if verbose:
                print(f"\n" + "#"*28 + " 2º Network Response " + "#"*28 + "\n")
                self._print_matrix(network_response)
                print(f"Error for this cycle is {self.loss_function(working_y, network_response)}")
            
    
    def backpropagation_batch(self, input_vectors, y_s, steps = 1000, learning_rate = 0.005, verbose=False):
        """
        input_vectors: list of input vectors [[x1_1, x1_2, ...], [x2_1, x2_2, ...], ...]
        y_s: list of corresponding labels [[y1_1, y1_2, ...], [y2_1, y2_2, ...], ...]
        """
        # Size of the batch
        batch_size = len(input_vectors)
        
        # Prepare y_s
        working_y_s = [self._tranpose_matrix(y if isinstance(y, list) else [y]) for y in y_s]
        
        # print(f"\n" + "#"*25 + " Init Backprop Matrixes " + "#"*25 + "\n")
        # Built backprop matrix_structure
        self._build_backprop_matrix()
        
        for step in range(steps):
            
            # Restart stroing matrixes
            store_weight_matrix_for_each_step = []
            store_bias_matrix_for_each_step = []
            
            for input_vector, working_y in zip(input_vectors, working_y_s):
                
                # Use the network and compute a prediction
                network_response = self.prediction([input_vector])
                
                # Start by performing backpropagation in the current weights and activations
                self._backpropagation(working_y)
                
                # Add the loss derivative w.r.t weights and bias matrixes to the storage
                store_weight_matrix_for_each_step.append(self._copy_structure(self.loss_deriv_of_weights))
                store_bias_matrix_for_each_step.append(self._copy_structure(self.loss_deriv_of_bias))
                
                # Clean Matrixes 
                for matrix in self.loss_deriv_of_weights:
                    self._fill_with_none(matrix)
                    
                for matrix in self.loss_deriv_of_bias:
                    self._fill_with_none(matrix)
            
            # Will hold the averaged weights and bias per layer
            averaged_weights = []
            averaged_bias = []
            
            for layer in range(self.n_layers - 1):
                # Collect all weight matrices for this layer across the batch
                matrices_to_average = [ex[layer] for ex in store_weight_matrix_for_each_step]
                averaged_weights.append(self._average_matrices(matrices_to_average))

                # Collect all bias matrices for this layer across the batch
                matrices_to_average_bias = [ex[layer] for ex in store_bias_matrix_for_each_step]
                averaged_bias.append(self._average_matrices(matrices_to_average_bias))
                
            # Now averaged_weights and bias has one matrix per layer, ready to update your network
            
            adjusted_weight = []
            adjusted_bias = []
        
            for matrix in averaged_weights:
                adjusted_weight.append(self._product_by_scalar(learning_rate, matrix))
            
            for bias in averaged_bias:
                adjusted_bias.append(self._product_by_scalar(learning_rate, bias))
            
            result_weight_matrix = []
            result_bias_matrix = []
            
            for j in range(self.n_layers -1):
                result_weight_matrix.append(self._subtract_matrices(self.weights[j],adjusted_weight[j]))
                result_bias_matrix.append(self._subtract_matrices(self.bias[j],adjusted_bias[j]))
            
            self.weights = self._copy_structure(result_weight_matrix)
            self.bias = self._copy_structure(result_bias_matrix)
                
            # Compute batch error for monitoring
            
            total_error = 0
            
            for x, y in zip(input_vectors, y_s):
    
                # Compute network prediction
                pred = self.prediction([x])
    
                # Compute loss for this example
                error = self.loss_function(pred, self._tranpose_matrix(y))
    
                # Add to total
                total_error += error

            # Average over batch
            batch_error = total_error / batch_size
            if verbose:
                print(f"\n{'#'*20} Network Response after cycle: {step} {'#'*20}\n")
                print(f"Batch Error for this cycle: {batch_error:.6f}\n")
    