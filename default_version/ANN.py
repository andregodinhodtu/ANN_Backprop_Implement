##################################
# Build Aritificial Neural Network
##################################

class ANN():
    
    ACTIVATION_FUNCTIONS = {
        "relu": {
            "func": lambda x: max(0, x),
            "deriv": lambda x: 1 if x > 0 else 0
        }
    }
    LOSS_FUNCTIONS = {
        "MSE": {
            "func": lambda x, y: sum(
                (x[i][0] - y[i][0]) ** 2 for i in range(len(x))
            ),
            "deriv": lambda x, y: 2 * (x - y)
        }
    }
    
    def __init__(self, n_layers, n_neurons_each_layer, activation_function, loss_function):
        
        # Input Confirmation
        if not isinstance(n_layers, int):
            raise TypeError("n_layers must be an Integer")
        if not isinstance(n_neurons_each_layer, list):
            raise TypeError("n_neurons_each_layer should be a list")
        if not all(isinstance(n, int) for n in n_neurons_each_layer):
            raise TypeError("All elements in n_neurons_each_layer must be integers")
        
        
        # Value checks
        if n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if len(n_neurons_each_layer) != n_layers:
           raise ValueError("Length of n_neurons_each_layer must equal n_layers")    
        if activation_function not in self.ACTIVATION_FUNCTIONS:
           raise ValueError(f"Unknown activation function: {activation_function}")
            
            
        # Initial settings obligatory
        self.n_layers =  n_layers
        self.n_neurons_each_layer =  n_neurons_each_layer
        self.activation_function = self.ACTIVATION_FUNCTIONS[activation_function]["func"]
        self.loss_function = self.LOSS_FUNCTIONS[loss_function]["func"]
        
        # Parameters of the Network to build
        self.weights = []
        self.bias = []
        
        # Build the ANN
        self._build_ANN()
        
        # Results in each step
        self.z_s = []
        self.a_s = []
        
        # Backpropagation
        self.loss_deriv_of_weights = []
        self.loss_deriv_of_bias = []
        self.delta_s = [[] for x in range(n_layers)]
        self.activation_function_deriv = self.ACTIVATION_FUNCTIONS[activation_function]["deriv"]
        self.loss_function_deriv = self.LOSS_FUNCTIONS[loss_function]["deriv"]
        
        
    def _print_matrix(self, matrix):
        for row in matrix:
            print("\t".join(
                f"{val:8.3f}" if val is not None else f"{'None':>8}"
                for val in row
            ))
        pass
        
    def _tranpose_matrix(self,matrix):
        """ Helper function to transpose a matrix"""
        # Number of rows and columns
        n_row = len(matrix)
        n_col = len(matrix[0])
    
        # Initialize output matrix
        matrix_T = []
    
        # Change positions and build new row
        for i in range(n_col):
            new_row= []
            for j in range(n_row):
                new_row.append(matrix[j][i])
        
            # Append new row to final matrix
            matrix_T.append(new_row)
            
        return matrix_T
        
    def _product(self,matrix_a,matrix_b):
        """ Matrix Multiplication of matrixes stored as lists of lists"""
    
        # Number of rows
        n_row_a = len(matrix_a)
        n_row_b = len(matrix_b)
    
        # Number of cols
        n_col_a = len(matrix_a[0])
        n_col_b = len(matrix_b[0])
    
        # Check if number of cols in A is the same of rows in B to perform product
        if n_col_a != n_row_b:
            raise ValueError("Product not possible, different number of cols in A and rows in B")
        
        # Initialize output matrix
        output_matrix = []
    
        # Transposed of B, useful for row * col multiplication      
        matrix_b_T = self._tranpose_matrix(matrix_b)
    
        # Running trough the calculation of each cell for the new matrix
        for row_a in matrix_a:
            output_row = []
            for col_b in matrix_b_T:
                output_row.append(sum([x * y for x, y in zip(row_a, col_b)]))
            
            output_matrix.append(output_row)

        return output_matrix
        
    def _product_by_scalar(self, scalar, matrix):
        """Multiply every element of a matrix by a scalar"""
    
        output_matrix = []
    
        for row in matrix:
            output_row = []
            for value in row:
                output_row.append(scalar * value)
            output_matrix.append(output_row)
    
        return output_matrix
        
    def _subtract_matrices(self, matrix_a, matrix_b):
        """Subtract two matrices element-wise (matrix_a - matrix_b)"""
    
        # Check dimensions
        if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
            raise ValueError("Matrices must have the same dimensions")
    
        # Element-wise subtraction
        result = []
        for row_a, row_b in zip(matrix_a, matrix_b):
            result.append([a - b for a, b in zip(row_a, row_b)])
    
        return result
        
        
    def _add_bias(self, z, b):
        """
        Add bias vector b to pre-activation z.
        """
        return [[z[i][0] + b[i][0]] for i in range(len(z))]
        
    def _apply_activation(self, vector):
        """
        Apply self.activation_function element-wise to a column vector,
        keeping it as a vertical vector.
    
        vector: list of lists [[x1], [x2], ...] or numpy column vector
        """
        return [[self.activation_function(x[0])] for x in vector]
        
        
    def _build_ANN(self):
        """ 
        Create the weights and biases
        """
        print(f"The builder will create a ANN with:")
        
        # Just checking the input
        for i, n_neurons in zip(range(self.n_layers), self.n_neurons_each_layer):
            print(f" - In the {i} layer with {n_neurons} neurons")
            
        print("\nBuild matrix and weights:\n")
        
        # Build the structure of the ANN    
        for i in range(self.n_layers - 1):
            print("-" * 40)
            # Weight matrix: next_layer_neurons x current_layer_neurons
            weight_matrix = [[1 for x in range(self.n_neurons_each_layer[i])] for j in range(self.n_neurons_each_layer[i+1])]
            self.weights.append(weight_matrix)
            print(f"Weights in {i}")
            print(self.weights)

            # Bias vector: next_layer_neurons x 1
            bias_vector = [[1] for _ in range(self.n_neurons_each_layer[i+1])]
            self.bias.append(bias_vector)

            print(f"Weight matrix between layer {i} and layer {i+1}:")
            self._print_matrix(weight_matrix)
            print(f"Bias Vector between layer {i} and layer {i+1}:")
            self._print_matrix(bias_vector)
        
    def prediction(self, input_vector):
        """
        Make an example pass through the ANN
        """

        # Convert input to column vector
        working_vector = self._tranpose_matrix(input_vector)
        
        self.a_s = [working_vector]  # store input as activation
        self.z_s = [working_vector]  # input_vector + list for pre-activations

        # Forward pass through each layer
        for i in range(self.n_layers - 1):
            W = self.weights[i]
            b = self.bias[i]    

            # z = W @ a + b
            z = self._product(W, working_vector)
            z = self._add_bias(z, b)
            self.z_s.append(z)  # store pre-activation

            # Apply activation
            a = self._apply_activation(z)
            self.a_s.append(a)   # store activation

            # Set input for next layer
            working_vector = a

        return working_vector  # output of last layer
        
    
    def _build_backprop_matrix(self):
        """ 
        Creates List structure that stores the derivatives of the Loss in function
        of the weights and the biasas 
        """
            
        print("\nBuilding Backprop matrix for Derivative of Loss on weights and Bias:\n")
        
        # Build the structure of the ANN    
        for i in range(self.n_layers - 1):
            
            weight_matrix = [[None for x in range(self.n_neurons_each_layer[i])] for j in range(self.n_neurons_each_layer[i+1])]
            self.loss_deriv_of_weights.append(weight_matrix)
           
            bias_vector = [[None] for x in range(self.n_neurons_each_layer[i+1])]
            self.loss_deriv_of_bias.append(bias_vector)

            print(f"Weight matrix of Loss derivative between layer {i} and layer {i+1}:")
            self._print_matrix(weight_matrix)
            print(f"Bias Matrix of Loss derivative between layer {i} and layer {i+1}:")
            self._print_matrix(bias_vector)
    
    
    def _backpropagation(self, y):
        
        # for eah layer i want to run trough all the values of parameters in that layer
        #that means running trough each next layer neurosn and inside for each layer neuron
        
        # First we will compute all the delta
        
        working_y = y
    
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
                    deriv_activation = self.activation_function_deriv(z)
                    
                    delta = deriv_loss * deriv_activation
                    self.delta_s[layer].append(delta)
                    
            else:
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
                    z = self.z_s[layer][neuron][0]
                    deriv_activation = self.activation_function_deriv(z)
                    
                    delta = sum_over_next_layer_neurons * deriv_activation
                    self.delta_s[layer].append(delta)
                    
                
                    
        # Now let´s fill in the  self.loss_deriv_of_weights and self.loss_deriv_of_bias
        for layer in range(self.n_layers-1):
            for neuron in range(self.n_neurons_each_layer[layer]):
                for next_layer_neuron in range(self.n_neurons_each_layer[layer+1]): 
                    self.loss_deriv_of_weights[layer][next_layer_neuron][neuron] = self.a_s[layer][neuron][0] * self.delta_s[layer+1][next_layer_neuron]
                    self.loss_deriv_of_bias[layer][next_layer_neuron][0] = self.delta_s[layer+1][next_layer_neuron]
                    
        
    def backprop_one_training_example(self, input_vector, y):
        """Apply Chain Rule
           We want to calculate the derivatives for the Lost in
           function of the Weights and the biases"""
        
        network_response = self.prediction(input_vector)
        working_y = self._tranpose_matrix(y)
        print(f"Working y: {working_y}")
        # loss = self.LOSS_FUNCTION[loss_function]["func"](network_response,working_y )
        # derivative = self.LOSS_FUNCTION[loss_function]["deriv"](network_response,working_y)
        
        print(f"\n" + "#"*25 + " Init Backprop Matrixes " + "#"*25 + "\n")
        self._build_backprop_matrix()
        
        
        print(f"\n" + "#"*30 + " First cycle " + "#"*30 + "\n")
        
        self._backpropagation(working_y)
        for i in range(self.n_layers - 1):

            print(f"Weight matrix of Loss derivative between layer {i} and layer {i+1}:")
            self._print_matrix(self.loss_deriv_of_weights[i])
            print(f"Bias Matrix of Loss derivative between layer {i} and layer {i+1}:")
            self._print_matrix(self.loss_deriv_of_bias[i])
            
        # Change weights 
        
        learning_rate = 0.01
        adjusted_weight = []
        adjusted_bias = []
        
        for matrix in self.loss_deriv_of_weights:
            adjusted_weight.append(self._product_by_scalar(learning_rate, matrix))
            
        for bias in self.loss_deriv_of_bias:
            adjusted_bias.append(self._product_by_scalar(learning_rate, bias))
            
            
        result_weight_matrix = []
        result_bias_matrix = []
        for i in range(self.n_layers -1):
            result_weight_matrix.append(self._subtract_matrices(self.weights[i],adjusted_weight[i]))
            result_bias_matrix.append(self._subtract_matrices(self.bias[i],adjusted_bias[i]))
            
        
        print(f"\n" + "#"*30 + " New parameters " + "#"*30 + "\n")
        
        for i in range(self.n_layers - 1):

            print(f"Weight matrix of Loss derivative between layer {i} and layer {i+1}:")
            self._print_matrix(result_weight_matrix[i])
            print(f"Bias Matrix of Loss derivative between layer {i} and layer {i+1}:")
            self._print_matrix(result_bias_matrix[i])
            
        
        
        

test_n_layers = 3
test_n_neurons_each_layer = [3,2,3]
test_nn = ANN(test_n_layers,
             test_n_neurons_each_layer,
             "relu",
             "MSE")
             
test_prediction = [[1,1,2]]
working_y = [[1,2,3]]


test_nn.backprop_one_training_example(test_prediction, working_y)