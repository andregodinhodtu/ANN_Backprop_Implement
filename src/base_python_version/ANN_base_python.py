##################################
# Build Aritificial Neural Network
##################################
import random
import math
from ANN_layer_base_python import ANN_Layer_base_python


class ANN_base_python():
 
    LOSS_FUNCTIONS = {
        "MSE": {
            "func": lambda x, y: sum(
                (x[i][0] - y[i][0]) ** 2 for i in range(len(x))
            ),
            "deriv": lambda x, y: 2 * (x - y)
        },
        "BinaryCrossEntropy" : {
            "func": lambda x, y: sum( - (y[i][0] * math.log(max(x[i][0], 1e-15)) +
                                    (1 - y[i][0]) * math.log(max(1 - x[i][0], 1e-15)))
                                    for i in range(len(x))) / len(x),  # divide by N
            "deriv": lambda x, y: (x - y) / (x * (1 - x) + 1e-15)
        }
    }
        
    def __init__(self, n_layers, n_neurons_each_layer, activation_hidden,
                 activation_output, loss_function, seed=None):
        """
        Build a feedforward neural network with n_layers.
        Parameters:
        -----------
        n_layers : int
            Number of layers (including input, hidden layers and output).
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
        if activation_hidden not in ANN_Layer_base_python.ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unknown hidden activation: {activation_hidden!r}. "
                f"Choose from {list(ACTIVATION_FUNCTIONS)}"
            )
        if activation_output not in ANN_Layer_base_python.ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unknown output activation: {activation_output!r}. "
                f"Choose from {list(ACTIVATION_FUNCTIONS)}"
            )
            
        if loss_function == "BinaryCrossEntropy" and activation_output != "sigmoid":
               raise ValueError(
                   "BinaryCrossEntropy requires sigmoid output activation. "
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
        self.rng = random.Random(seed)
        
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
            act = self.activation_output if i == self.n_layers - 2 else self.activation_hidden
            
            # Create ANN_Layer
            layer = ANN_Layer_base_python(
                n=i+1,
                n_neurons_input=n_input,
                n_neurons_output=n_output,
                activation_function=act
            )
            
            # Initialize weights and biases
            layer.initialize_weights_bias(self.rng)
            
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
        Compute delta values for each layer (backward pass).
        Stores them in each layer's `.delta` attribute.

        Parameters:
        -----------
        y : list of lists
            Target output column vector (shape: n_output x 1).
        """
        # --- Shape check on y ---
        expected = self.layers[-1].n_neurons_output
        if len(y) != expected:
            raise ValueError(
                f"y has {len(y)} elements, expected {expected} (output layer size)."
            )

        loss_deriv = self.LOSS_FUNCTIONS[self.loss_function]["deriv"]

        # Iterate layers from output back to input
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            # --- Shared setup ---
            layer.compute_activation_derivatives()
            layer.delta = []

            is_output = (i == len(self.layers) - 1)

            for j in range(layer.n_neurons_output):
                # --- The one thing that differs: where upstream signal comes from ---
                if is_output:
                    # dL/da for output neuron j
                    upstream = loss_deriv(layer.a_s[j][0], y[j][0])
                else:
                    # Sum_k delta_next[k] * W_next[k][j]
                    next_layer = self.layers[i + 1]
                    upstream = sum(
                        next_layer.delta[k] * next_layer.weights[k][j]
                        for k in range(next_layer.n_neurons_output)
                    )

                # --- Shared finish ---
                delta = upstream * layer.activation_derivatives[j][0]   # ← [0] unwraps the column-vector cell
                layer.delta.append(delta)

    def compute_gradients_sample(self, input_vector, target):
        """
        Compute gradients (dweights, dbiases) for a single training sample.

        Parameters:
        -----------
        input_vector : list of lists
            Input column vector (shape: n_input x 1).
        target : list of lists
            Target output column vector (shape: n_output x 1).
        """
        # Forward pass populates a_s on each layer
        self.prediction(input_vector)

        # Backward pass populates delta on each layer
        self._compute_deltas(target)

        # Compute per-parameter gradients
        for i, layer in enumerate(self.layers):
            # Activations entering this layer
            prev_activations = input_vector if i == 0 else self.layers[i - 1].a_s

            # dW[j][k] = delta[j] * a_prev[k]   (shape: n_out x n_in)
            layer.dweights = [
                [layer.delta[j] * prev_activations[k][0]
                 for k in range(layer.n_neurons_input)]
                for j in range(layer.n_neurons_output)
            ]

            # db[j] = delta[j]   (shape: n_out x 1, column-vector to match biases)
            layer.dbiases = [[layer.delta[j]] for j in range(layer.n_neurons_output)]
    
    def compute_gradients_batch(self, batch_inputs, batch_targets):
        """
        Compute averaged gradients (dweights, dbiases) over a mini-batch.

        For each (x, y) sample in the batch:
          1. Run the forward pass.
          2. Backpropagate to get per-sample gradients.
          3. Accumulate them.

        After all samples are processed, divide by the batch size and store the
        averaged gradients on each layer's `dweights` and `dbiases` attributes,
        ready to be consumed by `update_parameters`.

        Parameters:
        -----------
        batch_inputs : list of (list of lists)
            Mini-batch of input column vectors, each of shape (n_input, 1).
        batch_targets : list of (list of lists)
            Mini-batch of target column vectors, each of shape (n_output, 1).
        """
        # --- Input checks ---
        if len(batch_inputs) != len(batch_targets):
            raise ValueError(
                f"batch_inputs has {len(batch_inputs)} elements, "
                f"batch_targets has {len(batch_targets)}. They must match."
            )
        batch_size = len(batch_inputs)
        if batch_size == 0:
            raise ValueError("Batch is empty; nothing to compute.")

        # --- 1. Initialize zero accumulators (column-vector format throughout) ---
        accum_dweights = [
            [[0.0 for _ in range(layer.n_neurons_input)]
             for _ in range(layer.n_neurons_output)]
            for layer in self.layers
        ]
        accum_dbiases = [
            [[0.0] for _ in range(layer.n_neurons_output)]
            for layer in self.layers
        ]

        # --- 2. Accumulate per-sample gradients ---
        for x, y in zip(batch_inputs, batch_targets):
            self.compute_gradients_sample(x, y)

            for i, layer in enumerate(self.layers):
                for j in range(layer.n_neurons_output):
                    for k in range(layer.n_neurons_input):
                        accum_dweights[i][j][k] += layer.dweights[j][k]
                    accum_dbiases[i][j][0] += layer.dbiases[j][0]  
                    
        # --- 3. Average and store on each layer ---
        for i, layer in enumerate(self.layers):
            layer.dweights = [
                [accum_dweights[i][j][k] / batch_size
                 for k in range(layer.n_neurons_input)]
                for j in range(layer.n_neurons_output)
            ]
            layer.dbiases = [
                [accum_dbiases[i][j][0] / batch_size]
                for j in range(layer.n_neurons_output)
            ]    
        
    def train(self, X, Y, epochs=10, learning_rate=0.01, batch_size=1, 
              verbose=True, lr_decay=0.95, decay_every=20, l2_lambda = 0):  # ← add these params
        """
        Train the ANN using mini-batch gradient descent.

        Parameters:
        -----------
        X : list of input column vectors
            Shape: (n_samples, n_input, 1)
        Y : list of target column vectors
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
        current_lr = learning_rate  # ← track current lr

        for epoch in range(1, epochs + 1):
        
            # Decay learning rate every N epochs
            if epoch > 1 and (epoch - 1) % decay_every == 0:
                current_lr *= lr_decay
                if verbose:
                    print(f"  [LR decayed to {current_lr:.6f}]")

            indices = list(range(n_samples))
            random.shuffle(indices)
            X_shuffled = [X[i] for i in indices]
            Y_shuffled = [Y[i] for i in indices]

            epoch_loss = 0.0

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_X = X_shuffled[start_idx:end_idx]
                batch_Y = Y_shuffled[start_idx:end_idx]

                self.compute_gradients_batch(batch_X, batch_Y)

                for layer in self.layers:
                    layer.update_parameters(current_lr, l2_lambda)  # ← use current_lr

                for x_sample, y_sample in zip(batch_X, batch_Y):
                    pred = self.prediction(x_sample)
                    loss = self.LOSS_FUNCTIONS[self.loss_function]["func"](pred, y_sample)
                    epoch_loss += loss

            epoch_loss /= n_samples

            if verbose:
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}")
        