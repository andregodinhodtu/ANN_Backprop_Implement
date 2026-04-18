import random
import numpy as np
from ANN_layer import ANN_Layer

class ANN():

    """ANN algorithm with backpropagation made specifically for binary classification.
    This version utilizes NumPy for maximum efficiency and clarity. """

    # -----------------------------------------------------------------------------

    # binary cross entropy for binary classification problem
    # y = true binary labels
    # y_pred = predicted probabilities after last activation (output)

    # loss
    @staticmethod
    def binary_cross_entropy(y, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # gradient
    @staticmethod
    def binary_cross_entropy_deriv(y, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return (-y / y_pred + (1 - y) / (1 - y_pred)) / y.size


    # --------------------------------------------------------------------------
    # INITIALIZATION


    def __init__(self, n_layers, n_neurons_each_layer, activation_hidden="relu",
                 activation_output="sigmoid", loss_function="mse"):

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

        self.layers = []
        self._build_ANN()
        

    def _build_ANN(self):
        """Private method to construct the layers of the network with Numpy-based ANN Layer."""
        for i in range(self.n_layers -1):
            # Number of inputs for this layer
            n_input = self.n_neurons_each_layer[i]
            # Number of neurons in this layer
            n_output = self.n_neurons_each_layer[i+1]
            # Choose activation
            act = self.activation_output if i == self.n_layers - 2 else self.activation_hidden
            
            # Create ANN_Layer
            layer = ANN_Layer(
                n=i+1,
                n_neurons_input=n_input,
                n_neurons_output=n_output,
                activation_function=act
            )
            
            # Initialize weights and biases (optional fixed seed)
            layer.initialize_weights_bias(seed=42 +i)
            
            # Add to layers list
            self.layers.append(layer)


    # ----------------------------------------------------------------------------
    # PREDICTION
            
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
    

    # -----------------------------------------------------------------------------
    # BACPROPAGATION

    # chain rule:
    # 
    #  delta of the node = deriv of the loss function for this node   * derivative of act.func(z of this node)
    # 
    #  derivative of a weight =  delta * output a of the connected node in a previous layer
    #  derivative of a bias = delta

       
    def _compute_deltas(self, y):
        """
        Compute delta values for each layer in the network for backpropagation.
        Stores them in each layer's `.delta` attribute.

        Parameters:
        -----------
        y : list of lists or np.array
            Target output column vector (shape: n_output x 1)
        """

        y = np.array(y)

        # first: output layer 
        output_layer = self.layers[-1]
        output_layer.compute_activation_derivatives()
        # activations of output layer
        a = output_layer.a_s

        # compute loss derivative
        if self.loss_function == "mse":
            loss_deriv = 2*(a - y)
        elif self.loss_function in ["bse", "binarycrossentropy", "binary_cross_entropy"]:
            loss_deriv = self.binary_cross_entropy_deriv(y, a)
        else: 
            raise ValueError("Unknown loss function.")
        
        # save delta
        output_layer.delta = loss_deriv * output_layer.activation_derivatives
        
        # backwards loop through hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i+1]
            layer.compute_activation_derivatives()

            # next_layer.weights: shape (n_neurons_next_layer, n_neurons_this_layer)
            # next_layer.delta: shape (n_neurons_next_layer, 1)
            # next_layer.weights[k, j] = weight of neuron j from this layer to neuron k in next
            # sum (w_this-next * delta_next) for all neurons

            weighted_sum = np.dot(next_layer.weights.T, next_layer.delta)
            layer.delta = weighted_sum * layer.activation_derivatives
            print(f"{i}: {layer.delta}")
        




    
# test 

ann = ANN(5, [10, 3, 2, 2, 1]) #activation_hidden="sigmoid")
x = ann.prediction([[1],[2], [1],[2],[1],[2],[1],[2],[6],[3] ])
print(x)
ann._compute_deltas([[100]])