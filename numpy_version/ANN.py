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
    # PREDICTION (single sample) as a reference
            
    def prediction(self, input_vector):


        x = np.array(input_vector)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            # shape stays (n_features, 1)

        # foward pass by calling layer
        for layer in self.layers:
            x = layer(x)
        return x
    


    # FORWARD PASS - prediction over a full batch at once

    # X : np.ndarray, shape (n_samples, n_input, 1) (what train() takes)
    # transposed to (n_input, n_samples)

    def _forward_batch(self, input_batch):

        """
        Forward pass for the entire mini-batch at once. 
        X : (n_samples, n_input, 1)
        
        each layer stores a_s in shape (n_neurons, n_)"""

        # get shape (n_input, n_samples)
        x = input_batch[:, :, 0].T
        # foward pass by calling layer
        for layer in self.layers:
            x = layer(x) # works with any 2d input
        return x

    

    # -----------------------------------------------------------------------------
    # BACPROPAGATION

    # chain rule:
    # 
    #  delta of the node = deriv of the loss function for this node   * derivative of act.func(z of this node)
    # 
    #  derivative of a weight =  delta * output a of the connected node in a previous layer
    #  derivative of a bias = delta

       
    def _compute_deltas(self, y_batch):
        """
        Compute delta values for each layer in the network for backpropagation.
        Stores them in each layer's `.delta` attribute.

        Parameters:
        -----------
        y_batch : (batch_size, n_output, 1)
        """

        # shape
        y = y_batch[:, :, 0].T

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

            # (n_this, batch_size) = (n_this, n_next) @ (n_next, batch_size)
            weighted_sum = np.dot(next_layer.weights.T, next_layer.delta)
            layer.delta = weighted_sum * layer.activation_derivatives
        
    # compute 1 sample (func used only as a reference)
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
        Computes average gradients (dweights and dbiases) for a batch of samples using NumPy vectorized computation.

        batch_inputs: shape (batch_size, n_input, 1)
        batch_targets: shape (batch_size, n_output, 1)
        """

        batch_inputs  = np.array(batch_inputs)
        batch_targets = np.array(batch_targets)
        batch_size    = len(batch_inputs)

        # forward and backward pass over entire batch
        self._forward_batch(batch_inputs)
        self._compute_deltas(batch_targets)

        # gradient calculation

        # dW = (1/B) * delta @ prev_a.T
        # shape: (n_out, n_in), (n_out, B) @ (B, n_in)
        # depend on input values + loss

        # db = average of deltas over the batch
        # depend only on loss

        for i, layer in enumerate(self.layers):
            if i == 0:
                prev_a = batch_inputs[:, :, 0].T   # (n_features, batch_size)
            else:
                prev_a = self.layers[i - 1].a_s    # (n_in, batch_size)

            layer.dweights = np.dot(layer.delta, prev_a.T) / batch_size
            layer.dbiases  = layer.delta.mean(axis=1, keepdims=True)



        




        

# -----------------------------------------------------------------------------      
# TRAINING

    # mini-batch gradient descent

    def train(self, X, Y, epochs=10, learning_rate=0.01, batch_size=1, 
              verbose=True, lr_decay=0.95, decay_every=20, l2_lambda=0):
        
        # X, Y - whole training dataset

        """
        Train the ANN using mini-batch gradient descent.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_input, 1)
        Y : np.ndarray, shape (n_samples, n_output, 1)
        epochs : int
        learning_rate : float
        batch_size : int
        verbose : bool
        lr_decay : float   - multiplicative decay factor
        decay_every : int  - decay LR every N epochs
        l2_lambda : float  - L2 regularization strength (0 = off)
        """
        
        X = np.array(X)
        Y = np.array(Y)
        n_samples = len(X)
        current_lr = learning_rate

        for epoch in range(epochs):

            # LR decay
            if epoch != 0 and epoch % decay_every == 0:
                current_lr *= lr_decay
                if verbose:
                    print(f"  [LR decayed to {current_lr:.6f}]")

            # shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            # reset loss
            epoch_loss = 0.0

            # mini-batch loop
            for start in range(0, n_samples, batch_size):
                # get batch
                end = min(start + batch_size, n_samples)
                batch_X = X_shuffled[start:end]
                batch_Y = Y_shuffled[start:end]

                # compute gradients of the batch
                self.compute_gradients_batch(batch_X, batch_Y)

                # update layer parameters and L2 if set
                for layer in self.layers:
                    if l2_lambda > 0: # apply regularization (penalizes large weights)ž

                        # weight decay
                        # j_reg = j + reg
                        # reg = (l2_lambda / 2) * weights**2
                        # derivative of reg = l2_lambda * weights
                        layer.dweights += l2_lambda * layer.weights

                        layer.weights -= current_lr * layer.dweights
                        layer.biases  -= current_lr * layer.dbiases

            
            if verbose:
            # epoch loss - vectorized

                # all samples -> predictions
                all_preds = np.array([self.prediction(x) for x in X])

                # ensure same shape of true labels and predictions
                Y_flat = Y.reshape(n_samples, -1)
                P_flat = all_preds.reshape(n_samples, -1)

                # calculate loss
                if self.loss_function == "mse":
                    epoch_loss = np.mean((P_flat - Y_flat) ** 2)
                elif self.loss_function in ["bse", "binarycrossentropy", "binary_cross_entropy"]:
                    epoch_loss = self.binary_cross_entropy(Y_flat, P_flat)
                else:
                    raise ValueError("Unknown loss function")

                if verbose:
                    print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f} - LR: {current_lr:.6f}")


                













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