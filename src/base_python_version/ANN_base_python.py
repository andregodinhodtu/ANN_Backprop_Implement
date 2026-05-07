import random
import math
import os
from ANN_layer_base_python import ANN_Layer_base_python


class ANN_base_python():
    
    """ANN algorithm with backpropagation made specifically for binary classification.
    This version makes use of nested list structures from Core python."""
    
    LOSS_FUNCTIONS = {
        "mse": {
            "func":  lambda y_true, y_pred: (y_pred - y_true) ** 2,
            "deriv": lambda y_true, y_pred: 2 * (y_pred - y_true),
        },
        "binarycrossentropy": {
            "func":  lambda y_true, y_pred: -(
                y_true * math.log(max(y_pred, 1e-15)) +
                (1 - y_true) * math.log(max(1 - y_pred, 1e-15))
            ),
            "deriv": lambda y_true, y_pred:
                (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-15),
        },
    }
        
    def __init__(self, n_layers, n_neurons_each_layer, activation_hidden,
                 activation_output, loss_function, rng=None):
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
        seed : rng or None, optional
            random.Random for reproducible weight initialization.
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
                f"Choose from {list(ANN_Layer_base_python.ACTIVATION_FUNCTIONS)}"
            )
        if activation_output not in ANN_Layer_base_python.ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unknown output activation: {activation_output!r}. "
                f"Choose from {list(ANN_Layer_base_python.ACTIVATION_FUNCTIONS)}"
            )
            
        if loss_function == "binarycrossentropy" and activation_output != "sigmoid":
               raise ValueError(
                   "binarycrossentropy requires sigmoid output activation. "
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
        self.rng = rng
        
        # Option to have dedicated RNG so this network's randomness
        if self.rng is None:
            self.rng = random.Random()
        
        # Layers container
        self.layers = []
        
        # Build layers
        # In initialization network is built
        self._build_ANN()
        
    def _build_ANN(self):
        """
        Private method to construct the layers of 
        the network with Numpy-based ANN Layer.
        Called in __init__
        """
        
        # Creating each layer at a time
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
            
            # Initialize weights and biases (optional fixed seed)
            layer.initialize_weights_bias(self.rng)
            
            # Add to layers list
            self.layers.append(layer)
            
    def prediction(self, input_vector):
        """
        Make a forward pass through the entire ANN for a single sample.

        Parameters:
        input_vector : list of lists
            Input column vector, shape (n_input, 1).
            Must be a list of single-element lists, e.g. [[0.5], [0.2], [0.9]].

        Returns:
        list of lists
            Output of the last layer after activation, shape (n_output, 1).
        """
        # Type checks
        if not isinstance(input_vector, list):
            raise TypeError("input_vector must be a list of lists")
        if not all(isinstance(row, list) for row in input_vector):
            raise TypeError("All elements of input_vector must be lists (rows)")
        if not all(isinstance(val, (int, float)) for row in input_vector for val in row):
            raise TypeError("All values in input_vector must be ints or floats")

        # Value checks
        if len(input_vector) == 0:
            raise ValueError("input_vector cannot be empty")
        if any(len(row) != 1 for row in input_vector):
            raise ValueError("Each row in input_vector must contain exactly 1 element")

        # Dimension check against network's expected input size
        expected = self.n_neurons_each_layer[0]
        if len(input_vector) != expected:
            raise ValueError(
                f"input_vector has {len(input_vector)} features, "
                f"but the network expects {expected}."
            )

        # Forward pass through all layers
        working_vector = input_vector
        for layer in self.layers:
            # Use the layer's __call__ to do forward pass and activation
            working_vector = layer(working_vector)

        return working_vector
        
    def _compute_deltas(self, y):
        """
        Compute delta values for each layer (backward pass).
        Stores them in each layer's `.delta` attribute.

        Parameters:
        y : list of lists
            Target output column vector (shape: n_output x 1).
        """
        expected = self.layers[-1].n_neurons_output
        if len(y) != expected:
            raise ValueError(
                f"y has {len(y)} elements, expected {expected} (output layer size)."
            )

        # Iterate layers from output back to input
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            layer.compute_activation_derivatives()
            layer.delta = []

            is_output = (i == len(self.layers) - 1)

            # Special case: BCE + sigmoid output
            # The gradient (a - y)/(a*(1-a)) * a*(1-a) simplifies to (a - y).
            # Compute this directly to avoid catastrophic cancellation.
            if is_output and self.loss_function == "binarycrossentropy" \
                         and layer.activation_function == "sigmoid":
                for j in range(layer.n_neurons_output):
                    delta = layer.a_s[j][0] - y[j][0]
                    layer.delta.append(delta)
                # skip the generic path for this layer
                continue

            # Generic path
            loss_deriv = self.LOSS_FUNCTIONS[self.loss_function]["deriv"]

            for j in range(layer.n_neurons_output):
                if is_output:
                    upstream = loss_deriv(y[j][0], layer.a_s[j][0])
                else:
                    next_layer = self.layers[i + 1]
                    # averged sum with weights between layers and
                    # previous layers
                    upstream = sum(
                        next_layer.delta[k] * next_layer.weights[k][j]
                        for k in range(next_layer.n_neurons_output)
                    )
                
                # to get delta just multiply by the activations
                delta = upstream * layer.activation_derivatives[j]
                layer.delta.append(delta)
    
    def _save_parameters_snapshot(self):
        """
        Deep-copy current weights and biases of all layers.
        """
        return [(
            [row[:] for row in layer.weights],
            [row[:] for row in layer.biases]
        ) for layer in self.layers]

    def _restore_parameters_snapshot(self, saved):
        """Restore a previously saved weights/biases snapshot."""
        
        # assigning the saved to weigths and biases
        for layer, (w, b) in zip(self.layers, saved):
            layer.weights = [row[:] for row in w]
            layer.biases  = [row[:] for row in b]
       
    def compute_gradients_sample(self, input_vector, target):
        """
        Compute gradients (dweights, dbiases) for a single training sample.

        Parameters:
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
        batch_inputs : list of (list of lists)
            Mini-batch of input column vectors, each of shape (n_input, 1).
        batch_targets : list of (list of lists)
            Mini-batch of target column vectors, each of shape (n_output, 1).
        """
        # Input checks
        if len(batch_inputs) != len(batch_targets):
            raise ValueError(
                f"batch_inputs has {len(batch_inputs)} elements, "
                f"batch_targets has {len(batch_targets)}. They must match."
            )
        batch_size = len(batch_inputs)
        if batch_size == 0:
            raise ValueError("Batch is empty; nothing to compute.")

        # 1. Initialize zero accumulators (column-vector format throughout)
        accum_dweights = [
            [[0.0 for i in range(layer.n_neurons_input)]
             for j in range(layer.n_neurons_output)]
            for layer in self.layers
        ]
        accum_dbiases = [
            [[0.0] for i in range(layer.n_neurons_output)]
            for layer in self.layers
        ]

        # 2. Accumulate per-sample gradients
        for x, y in zip(batch_inputs, batch_targets):
            self.compute_gradients_sample(x, y)

            for i, layer in enumerate(self.layers):
                for j in range(layer.n_neurons_output):
                    for k in range(layer.n_neurons_input):
                        accum_dweights[i][j][k] += layer.dweights[j][k]
                    accum_dbiases[i][j][0] += layer.dbiases[j][0]  
                    
        # 3. Average and store on each layer
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
    
    def compute_loss(self, X, Y):
        """
        Compute the mean loss across a batch of samples.

        Parameters:
        X : list of (list of lists)
            Batch of input column vectors.
        Y : list of (list of lists)
            Batch of target column vectors, shape (n_output, 1) each.

        Returns:
        float
            Mean loss across the batch (averaged over samples and output neurons).
        """
        loss_func = self.LOSS_FUNCTIONS[self.loss_function]["func"]

        total_loss = 0.0
        for x, y in zip(X, Y):
            # column vector (n_output, 1)
            y_pred = self.prediction(x)
            n = len(y)
            # Mean per-neuron loss for this sample
            sample_loss = sum(
                loss_func(y[i][0], y_pred[i][0]) for i in range(n)
            ) / n
            total_loss += sample_loss

        return total_loss / len(X)
     
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
        X_train, Y_train : training data (lists of column vectors)
        X_val, Y_val     : validation data (lists of column vectors)
        epochs           : maximum number of epochs
        learning_rate    : initial learning rate
        batch_size       : mini-batch size
        lr_decay         : multiplicative factor applied every `decay_every` epochs
        decay_every      : LR decay frequency in epochs
        l2_lambda        : L2 regularization coefficient
        patience         : stop after this many epochs without val-loss improvement
        verbose          : print per-epoch progress

        Returns:
        history : dict with 'train_loss' and 'val_loss' lists per epoch
        """

        # Save hyperparameters for later (used by save_model)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.decay_every = decay_every
        self.l2_lambda = l2_lambda
        self.n_samples = len(X_train)

        # Early stopping state
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
            indices = list(range(len(X_train)))
            self.rng.shuffle(indices)
            X_shuffled = [X_train[i] for i in indices]
            Y_shuffled = [Y_train[i] for i in indices]

            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_X = X_shuffled[start:end]
                batch_Y = Y_shuffled[start:end]
                self.compute_gradients_batch(batch_X, batch_Y)
                for layer in self.layers:
                    layer.update_parameters(current_lr, l2_lambda)

            # Track losses for this epoch
            # Compute the loss on train and val and store them
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
        
    def save_model(self, output_filename, data_name, path):
        """
        Save model parameters in a consistent and replicable way.
        """

        if not isinstance(output_filename, str):
            raise TypeError("output_filename must be a string")
        if not isinstance(data_name, str):
            raise TypeError("data_name must be a string")

        save_dir = path
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, output_filename)

        with open(full_path, "w", encoding="utf-8") as file:
            # Metadata
            file.write(f">Model: {output_filename}\n")
            file.write(f">Data used to train: {data_name}\n")
            file.write(f">Number of samples: {self.n_samples}\n")
            file.write(f">Epochs: {self.epochs}\n")
            file.write(f">Learning rate: {self.learning_rate}\n")
            file.write(f">Batch size: {self.batch_size}\n")
            file.write(f">Learning rate decay: {self.lr_decay}\n")
            file.write(f">Decay every: {self.decay_every}\n")
            file.write(f">L2 lambda: {self.l2_lambda}\n")

            # Architecture
            arch_str = ",".join(str(n) for n in self.n_neurons_each_layer)
            file.write(f">N layers: {self.n_layers}\n")
            file.write(f">Architecture: {arch_str}\n")
            file.write(f">Activation hidden: {self.activation_hidden}\n")
            file.write(f">Activation output: {self.activation_output}\n")
            file.write(f">Loss function: {self.loss_function}\n")

            # Parameters per layer
            for i, layer in enumerate(self.layers):
                n_out = len(layer.weights)
                n_in  = len(layer.weights[0])

                file.write(f">Layer {i} weights: {n_out}x{n_in}\n")
                for row in layer.weights:
                    file.write(" ".join(f"{w:.10f}" for w in row) + "\n")

                file.write(f">Layer {i} biases: {n_out}x1\n")
                for row in layer.biases:
                    file.write(f"{row[0]:.10f}\n")

        print(f"Model saved to: {os.path.abspath(full_path)}")

    @classmethod
    def load_model(cls, filepath):
        """
        Reconstruct an ANN from a saved model file.
        """
        with open(filepath, "r", encoding="utf-8") as file:
            lines = [line.rstrip("\n") for line in file]

        # First pass: parse header lines into a dict
        headers = {}
        data_lines = []
        for line in lines:
            if line.startswith(">"):
                key, _, value = line[1:].partition(":")
                headers[key.strip()] = value.strip()
            data_lines.append(line)

        # Build the model from architecture info
        architecture = [int(n) for n in headers["Architecture"].split(",")]
        ann = cls(
            n_layers=int(headers["N layers"]),
            n_neurons_each_layer=architecture,
            activation_hidden=headers["Activation hidden"],
            activation_output=headers["Activation output"],
            loss_function=headers["Loss function"],
        )

        # Second pass: walk through lines and load weights/biases
        i = 0
        layer_idx = 0
        while i < len(data_lines):
            line = data_lines[i]

            if line.startswith(">Layer") and "weights" in line:
                # Example ">Layer 0 weights: 32x27"
                shape_str = line.split(":")[1].strip()
                n_out, n_in = (int(x) for x in shape_str.split("x"))

                # Read the next n_out lines as weight rows
                rows = []
                for j in range(n_out):
                    row = [float(v) for v in data_lines[i + 1 + j].split()]
                    rows.append(row)
                # Appending the weights to ann
                ann.layers[layer_idx].weights = rows  # list of lists
                i += 1 + n_out

            elif line.startswith(">Layer") and "biases" in line:
                shape_str = line.split(":")[1].strip()
                n_out, _ = (int(x) for x in shape_str.split("x"))

                biases = [[float(data_lines[i + 1 + j])] for j in range(n_out)]
                 # Appending the biases to ann
                ann.layers[layer_idx].biases = biases   # list of single-element lists
                i += 1 + n_out
                layer_idx += 1

            else:
                i += 1

        print(f"Model loaded from: {os.path.abspath(filepath)}")
        return ann