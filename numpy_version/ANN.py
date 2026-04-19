import random
import numpy as np
import data_prep_np
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
        return (-y / y_pred) + ((1 - y) / (1 - y_pred))


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
        output_layer.delta = (loss_deriv * output_layer.activation_derivatives).reshape(-1,1)
        
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
            layer.delta = (weighted_sum * layer.activation_derivatives).reshape(-1,1) # ensure correct shape
        

    def compute_gradients_sample(self, input_vector, target):
        """
        Computes gradients (dweights and dbiases) for a single training sample using NumPy.
        
        input_vector: input column vector (shape: n_input x 1, as np.array or list of lists)
        target: target output column vector (shape: n_output x 1, as np.array or list of lists)
        """

        # input must be np.array
        input_vector = np.array(input_vector)

        # forward pass
        self.prediction(input_vector)

        # backward pass
        self._compute_deltas(target)

        # gradients for each layer
        for i, layer in enumerate(self.layers):
            # determine previous activations
            if i == 0:
                prev_a = input_vector
            else:
                prev_a = self.layers[i-1].a_s

            # get gradients
            layer.dweights = np.dot(layer.delta, prev_a.T) # W = delta * prev_a
            layer.dbiases = layer.delta # B = delta
           # print(layer.dweights.shape)
           # print(layer.dbiases.shape)
            

    def compute_gradients_batch(self, batch_inputs, batch_targets):

        """
        Computes average gradients (dweights and dbiases) for a batch of samples using NumPy.

        batch_inputs: shape (batch_size, n_input, 1)
        batch_targets: shape (batch_size, n_output, 1)
        """

        # gradient storage for all layers
        accum_dweights = [np.zeros_like(layer.weights) for layer in self.layers]
        accum_dbiases = [np.zeros_like(layer.biases) for layer in self.layers]
        
        # compute gradients per layer
        for x, y in zip(batch_inputs, batch_targets):
            self.compute_gradients_sample(x, y)

            
            # save each layers dweights and dbiases
            for i, layer in enumerate(self.layers):
                accum_dweights[i] += layer.dweights
                accum_dbiases[i] += layer.dbiases

        # average results per layer
        batch_size = len(batch_inputs)
        for i, layer in enumerate(self.layers):
            layer.dweights = accum_dweights[i] / batch_size
            layer.dbiases = accum_dbiases[i] / batch_size
        

# -----------------------------------------------------------------------------      
# TRAINING

    # mini-batch gradient descent

    def train(self, X, Y, epochs=10, learning_rate=0.01, batch_size=1, 
              verbose=True, lr_decay=0.95, decay_every=20, l2_lambda = 0):
        
        # X, Y - whole training dataset

        """
        Train the ANN using mini-batch gradient descent.

        Parameters:
        -----------
        X : array of input column vectors
            Shape: (n_samples, n_input, 1)
        Y : array of target column vectors
            Shape: (n_samples, n_output, 1)
        epochs : int
            Number of full passes over the dataset
        learning_rate : float
            Step size for gradient descent
        batch_size : int
            Number of samples per mini-batch
        verbose : bool
            If True, prints loss per epoch
        """
           
        n_samples = len(X)
        current_lr = learning_rate

        












# test 


def test_compute_gradients_batch():
    # Load a small batch from your data file
    filename = "data/training_set.howlin"
    X, y = data_prep_np.parse_input(filename, start=0, end=4)  # 4 samples

    # Reshape X and y for ANN input: (batch_size, n_input, 1)
    batch_inputs = [X[i].reshape(-1, 1) for i in range(X.shape[0])]
    batch_targets = [np.array([[y[i]]]) for i in range(y.shape[0])]

    # Build a small ANN
    n_layers = 3
    n_neurons_each_layer = [X.shape[1], 5, 1]
    ann = ANN(n_layers, n_neurons_each_layer, activation_hidden="relu", activation_output="sigmoid", loss_function="binary_cross_entropy")

    # Compute gradients for the batch
    ann.compute_gradients_batch(batch_inputs, batch_targets)

    # Print gradients for each layer
    for i, layer in enumerate(ann.layers):
        print(f"Layer {i+1} dweights shape: {layer.dweights.shape}")
        print(f"Layer {i+1} dbiases shape: {layer.dbiases.shape}")
        #print(f"Layer {i+1} dweights (preview):\n{layer.dweights}")
        #print(f"Layer {i+1} dbiases (preview):\n{layer.dbiases}\n")

if __name__ == "__main__":
    test_compute_gradients_batch()