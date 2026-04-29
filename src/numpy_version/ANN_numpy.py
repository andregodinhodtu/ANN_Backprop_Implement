import random
import numpy as np
from pathlib import Path
from ANN_layer_numpy import ANN_Layer_numpy

class ANN_numpy():

    """ANN algorithm with backpropagation made specifically for binary classification.
    This version utilizes NumPy for maximum efficiency and clarity. """

    LOSS_FUNCTIONS = {
        "mse": {
            "func":  lambda y_true, y_pred: (y_pred - y_true) ** 2,
            "deriv": lambda y_true, y_pred: 2 * (y_pred - y_true),
        },
        "binarycrossentropy": {
            "func": lambda y_true, y_pred: -(
                y_true       * np.log(np.clip(y_pred,     1e-12, 1 - 1e-12)) +
                (1 - y_true) * np.log(np.clip(1 - y_pred, 1e-12, 1 - 1e-12))
            ),
            "deriv": lambda y_true, y_pred:
                (y_pred - y_true) /
                (np.clip(y_pred, 1e-12, 1 - 1e-12) * np.clip(1 - y_pred, 1e-12, 1 - 1e-12)),
        },
    }
    def __init__(self, n_layers, n_neurons_each_layer, activation_hidden,
                 activation_output, loss_function, seed = None):

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
        seed : int or None, optional
            Random seed for reproducible weight initialization.
            If None, randomness is non-deterministic.
        """
        # Input validation
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2 (input + output)")
        if len(n_neurons_each_layer) != n_layers:
            raise ValueError("Length of n_neurons_each_layer must equal n_layers")
        if any(n <= 0 for n in n_neurons_each_layer):
            raise ValueError("All layer sizes must be > 0")
        if activation_hidden not in ANN_Layer_numpy.ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unknown hidden activation: {activation_hidden!r}. "
                f"Choose from {list(ANN_Layer_numpy.ACTIVATION_FUNCTIONS)}"
            )
        if activation_output not in ANN_Layer_numpy.ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unknown output activation: {activation_output!r}. "
                f"Choose from {list(ANN_Layer_numpy.ACTIVATION_FUNCTIONS)}"
            )

        if loss_function == "binarycrossentropy" and activation_output != "sigmoid":
            raise ValueError(
                "binary_cross_entropy requires sigmoid output activation. "
                f"Got {activation_output!r}."
            )

        if loss_function not in self.LOSS_FUNCTIONS:
            raise ValueError(
                f"Unknown loss function: {loss_function!r}. "
                f"Choose from {list(self.LOSS_FUNCTIONS)}"
            )

        # Store config
        self.n_layers = n_layers
        self.n_neurons_each_layer = n_neurons_each_layer
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.loss_function = loss_function
        self.seed = seed

        # Dedicated RNG so this network's randomness is isolated from
        # the global random state (good practice).
        self.rng = np.random.default_rng(seed)

        # Layers container
        self.layers = []

        # Build layers
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
            layer = ANN_Layer_numpy(
                n=i+1,
                n_neurons_input=n_input,
                n_neurons_output=n_output,
                activation_function=act
            )
            
            # Initialize weights and biases (optional fixed seed)
            layer.initialize_weights_bias(self.rng)
            
            # Add to layers list
            self.layers.append(layer)
            
    def prediction(self, input_vector):
        """
        Forward pass through the entire network.

        Accepts a single sample or a batch and returns the network's output(s).
        Thanks to vectorization, the same code path handles both cases — the
        shape simply propagates through each layer.

        Parameters:
        -----------
        input_vector : np.ndarray
            Input to the network. Must be a 2D array with shape:
              - (n_features, 1)             → single sample as column vector
              - (n_features, batch_size)    → batch of samples as column vectors

        Returns:
        --------
        np.ndarray
            Network output, shape (n_output, batch_size). For a single sample,
            batch_size is 1, so the shape is (n_output, 1).
        """
        # --- Type check ---
        if not isinstance(input_vector, np.ndarray):
            raise TypeError("input_vector must be a numpy.ndarray")
        if not np.issubdtype(input_vector.dtype, np.number):
            raise TypeError("input_vector must contain numeric values")

        # --- Shape checks ---
        if input_vector.ndim != 2:
            raise ValueError(
                f"input_vector must be 2D with shape (n_features, batch_size); "
                f"got {input_vector.ndim}D shape {input_vector.shape}."
            )
        if input_vector.shape[1] == 0:
            raise ValueError("input_vector has 0 samples (batch_size must be >= 1).")

        # --- Dimension check against network's expected input size ---
        expected = self.n_neurons_each_layer[0]
        if input_vector.shape[0] != expected:
            raise ValueError(
                f"input_vector has {input_vector.shape[0]} features along axis 0, "
                f"but the network expects {expected}. "
                f"If your batch is shaped (batch_size, n_features), transpose with .T"
            )

        # --- Forward pass through all layers ---
        x = input_vector
        for layer in self.layers:
            # Use the layer's __call__ to do forward pass and activation
            x = layer(x)

        return x
    
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
        # use the selected loss function from the dictionary
        loss_func = self.LOSS_FUNCTIONS[self.loss_function.lower()]
        loss_deriv = loss_func["deriv"](y, a)
        
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
    
    def _save_parameters_snapshot(self):
        """Deep-copy current weights and biases of all layers."""
        weights = [layer.weights.copy() for layer in self.layers]
        biases  = [layer.biases.copy()  for layer in self.layers]
        return weights, biases

    def _restore_parameters_snapshot(self, saved):
        """Restore a previously saved weights/biases snapshot."""
        weights, biases = saved
        for layer, w, b in zip(self.layers, weights, biases):
            layer.weights = w.copy()
            layer.biases  = b.copy()
    
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

    def compute_loss(self, X, Y):
        """
        Compute the mean loss across a batch of samples.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features, 1) or (n_features, n_samples)
            Batch of input column vectors.
        Y : np.ndarray, shape (n_samples, n_output, 1) or (n_output, n_samples)
            Batch of target column vectors.

        Returns:
        --------
        float
            Mean loss across the batch (averaged over samples and output neurons).
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        # Normalize to (n_features, n_samples) and (n_output, n_samples)
        if X.ndim == 3:
            X = X[:, :, 0].T
        if Y.ndim == 3:
            Y = Y[:, :, 0].T

        Y_pred = self.prediction(X)              # shape (n_output, n_samples)
        loss_func = self.LOSS_FUNCTIONS[self.loss_function]["func"]
        return float(np.mean(loss_func(Y, Y_pred)))
    
    def train(self, X_train, Y_train, X_val, Y_val,
              epochs=200,
              learning_rate=0.01,
              batch_size=32,
              lr_decay=0.95,
              decay_every=20,
              l2_lambda=1e-4,
              patience=50,
              verbose=True):
        """
        Train the ANN with mini-batch gradient descent, LR decay, and early stopping.

        Uses the network's own RNG (self.rng, seeded in __init__) for shuffling.

        Parameters:
        -----------
        X_train, Y_train : np.ndarray, shape (n_samples, n_features, 1) / (n_samples, n_output, 1)
        X_val, Y_val     : same layout
        epochs           : maximum number of epochs
        learning_rate    : initial learning rate
        batch_size       : mini-batch size
        lr_decay         : multiplicative factor applied every `decay_every` epochs
        decay_every      : LR decay frequency in epochs
        l2_lambda        : L2 regularization coefficient
        patience         : stop after this many epochs without val-loss improvement
        verbose          : print per-epoch progress

        Returns:
        --------
        history : dict with 'train_loss' and 'val_loss' lists per epoch
        """
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        X_val   = np.asarray(X_val)
        Y_val   = np.asarray(Y_val)

        n_samples = len(X_train)

        # --- Save hyperparameters for later (used by save_model) ---
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.decay_every = decay_every
        self.l2_lambda = l2_lambda
        self.n_samples = n_samples

        # --- Early stopping state ---
        best_val_loss = float('inf')
        best_weights = None
        best_epoch = 0
        epochs_no_improve = 0
        history = {"train_loss": [], "val_loss": []}

        current_lr = learning_rate

        for epoch in range(1, epochs + 1):
            # LR decay
            if epoch > 1 and (epoch - 1) % decay_every == 0:
                current_lr *= lr_decay
                if verbose:
                    print(f"  [LR decayed to {current_lr:.6f}]")

            # Shuffle and run one epoch of mini-batch SGD
            indices = self.rng.permutation(n_samples)         # uses self.rng
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_X = X_shuffled[start:end]
                batch_Y = Y_shuffled[start:end]
                self.compute_gradients_batch(batch_X, batch_Y)
                for layer in self.layers:
                    layer.update_parameters(current_lr, l2_lambda)

            # Track losses for this epoch
            train_loss = self.compute_loss(X_train, Y_train)
            val_loss   = self.compute_loss(X_val, Y_val)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose:
                print(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                best_weights = self._save_parameters_snapshot()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"\n*** Early stopping at epoch {epoch} "
                              f"(best was epoch {best_epoch}) ***")
                    break

        # Restore best weights
        if verbose:
            print(f"\nRestoring best weights from epoch {best_epoch} "
                  f"(val loss: {best_val_loss:.6f})")
        self._restore_parameters_snapshot(best_weights)

        return history

    def save_model():
        pass
    
    @classmethod
    def load_model(cls, filepath):
        pass
    