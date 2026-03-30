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
    
    def __init__(self, n_layers, n_neurons_each_layer, activation_function):
        
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
        
        # Parameters of the Network to build
        self.weights = []
        self.bias = []
        
        # Build the ANN
        self._build_ANN()
        
        # Results in each step (I think we need to store them)
        self.z_s = []
        self.a_s = []
        
    def _print_matrix(self, matrix):
        for row in matrix:
            print("\t".join(f"{val:8.1f}" for val in row))
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
            
            # Weight matrix: next_layer_neurons x current_layer_neurons
            weight_matrix = [[1 for x in range(self.n_neurons_each_layer[i])] for j in range(self.n_neurons_each_layer[i+1])]
            self.weights.append(weight_matrix)

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
        self.z_s = []                # empty list for pre-activations

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
        
    def backpropagation(self,step):
        """Apply Chain Rule"""
        pass
                
    
        


test_n_layers = 6
test_n_neurons_each_layer = [4,2,5,3,5,1]
test_nn = ANN(test_n_layers,
             test_n_neurons_each_layer,
             "relu")
             
test_prediction = [[2,-5,3,10]]

test_run = test_nn.prediction(test_prediction)  
print(test_run)      