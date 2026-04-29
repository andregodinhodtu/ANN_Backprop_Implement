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
                 activation_output, loss_function, rng = None):

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

        # Dedicated RNG so this network's randomness is isolated from
        # the global random state (good practice).
        if self.rng is None:
            self.rng = np.random.default_rng()

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
       
    def _compute_deltas(self, y):
        """
        Compute delta values for each layer (backward pass).
        Stores them in each layer's `.delta` attribute.

        Parameters:
        -----------
        y : np.ndarray, shape (n_output, batch_size)
            Target output, in the same 2D layout as predictions.
            For a single sample, batch_size is 1.
        """
        # --- Shape check ---
        if y.ndim != 2:
            raise ValueError(
                f"y must be 2D with shape (n_output, batch_size); got shape {y.shape}."
            )
        expected = self.layers[-1].n_neurons_output
        if y.shape[0] != expected:
            raise ValueError(
                f"y has {y.shape[0]} elements along axis 0, expected {expected}."
            )

        # --- Output layer delta ---
        output_layer = self.layers[-1]
        output_layer.compute_activation_derivatives()
        a = output_layer.a_s                                  # (n_out, batch_size)

        loss_deriv = self.LOSS_FUNCTIONS[self.loss_function]["deriv"]
        output_layer.delta = loss_deriv(y, a) * output_layer.activation_derivatives

        # --- Hidden layer deltas (backward loop) ---
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            layer.compute_activation_derivatives()

            # (n_this, batch) = (n_this, n_next) @ (n_next, batch)
            weighted_sum = next_layer.weights.T @ next_layer.delta
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
        Compute gradients for a single training sample.

        Parameters:
        -----------
        input_vector : np.ndarray, shape (n_input, 1)
        target       : np.ndarray, shape (n_output, 1)
        """
        # --- Type checks ---
        if not isinstance(input_vector, np.ndarray):
            raise TypeError("input_vector must be a numpy.ndarray")
        if not isinstance(target, np.ndarray):
            raise TypeError("target must be a numpy.ndarray")
        if not np.issubdtype(input_vector.dtype, np.number):
            raise TypeError("input_vector must contain numeric values")
        if not np.issubdtype(target.dtype, np.number):
            raise TypeError("target must contain numeric values")

        # --- Shape checks (must be a single column vector) ---
        expected_in  = self.n_neurons_each_layer[0]
        expected_out = self.layers[-1].n_neurons_output
        if input_vector.shape != (expected_in, 1):
            raise ValueError(
                f"input_vector must have shape ({expected_in}, 1); "
                f"got {input_vector.shape}."
            )
        if target.shape != (expected_out, 1):
            raise ValueError(
                f"target must have shape ({expected_out}, 1); "
                f"got {target.shape}."
            )

        # Forward pass populates a_s on each layer
        self.prediction(input_vector)

        # Backward pass
        self._compute_deltas(target)

        # Per-parameter gradients (no batch averaging — single sample)
        for i, layer in enumerate(self.layers):
            prev_a = input_vector if i == 0 else self.layers[i - 1].a_s
            layer.dweights = layer.delta @ prev_a.T          # (n_out, n_in)
            layer.dbiases  = layer.delta                      # (n_out, 1)
            
    def compute_gradients_batch(self, batch_inputs, batch_targets):
        """
        Compute averaged gradients over a mini-batch.

        Parameters:
        -----------
        batch_inputs  : np.ndarray, shape (n_input,  batch_size)
        batch_targets : np.ndarray, shape (n_output, batch_size)
            Both arrays must use the "math world" layout where columns are samples.
            If your data is in (n_samples, n_features) row-layout, transpose with .T
            before calling.
        """
        # --- Type checks ---
        if not isinstance(batch_inputs, np.ndarray):
            raise TypeError("batch_inputs must be a numpy.ndarray")
        if not isinstance(batch_targets, np.ndarray):
            raise TypeError("batch_targets must be a numpy.ndarray")
        if not np.issubdtype(batch_inputs.dtype, np.number):
            raise TypeError("batch_inputs must contain numeric values")
        if not np.issubdtype(batch_targets.dtype, np.number):
            raise TypeError("batch_targets must contain numeric values")

        # --- Shape checks ---
        if batch_inputs.ndim != 2:
            raise ValueError(
                f"batch_inputs must be 2D with shape (n_input, batch_size); "
                f"got {batch_inputs.ndim}D shape {batch_inputs.shape}."
            )
        if batch_targets.ndim != 2:
            raise ValueError(
                f"batch_targets must be 2D with shape (n_output, batch_size); "
                f"got {batch_targets.ndim}D shape {batch_targets.shape}."
            )

        # --- Batch size consistency ---
        if batch_inputs.shape[1] != batch_targets.shape[1]:
            raise ValueError(
                f"batch_inputs has {batch_inputs.shape[1]} samples (axis 1), "
                f"batch_targets has {batch_targets.shape[1]}. They must match."
            )
        batch_size = batch_inputs.shape[1]
        if batch_size == 0:
            raise ValueError("Batch is empty; nothing to compute.")

        # --- Dimension checks against network's expected sizes ---
        expected_in  = self.n_neurons_each_layer[0]
        expected_out = self.layers[-1].n_neurons_output
        if batch_inputs.shape[0] != expected_in:
            raise ValueError(
                f"batch_inputs has {batch_inputs.shape[0]} features along axis 0, "
                f"but the network expects {expected_in}."
            )
        if batch_targets.shape[0] != expected_out:
            raise ValueError(
                f"batch_targets has {batch_targets.shape[0]} elements along axis 0, "
                f"but the network expects {expected_out}."
            )

        # --- Forward pass over entire batch ---
        self.prediction(batch_inputs)

        # --- Backward pass over entire batch ---
        self._compute_deltas(batch_targets)

        # --- Per-parameter gradients, averaged over the batch ---
        for i, layer in enumerate(self.layers):
            prev_a = batch_inputs if i == 0 else self.layers[i - 1].a_s

            layer.dweights = (layer.delta @ prev_a.T) / batch_size
            layer.dbiases  = layer.delta.mean(axis=1, keepdims=True)

    def compute_loss(self, X, Y):
        """
        Compute the mean loss across a batch of samples.

        Parameters:
        -----------
        X : np.ndarray, shape (n_features, n_samples)
        Y : np.ndarray, shape (n_output,   n_samples)
            Both arrays must use the "math world" layout where columns are samples.

        Returns:
        --------
        float
            Mean loss across the batch (averaged over samples and output neurons).
        """
        # --- Type checks ---
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray")
        if not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a numpy.ndarray")
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError("X must contain numeric values")
        if not np.issubdtype(Y.dtype, np.number):
            raise TypeError("Y must contain numeric values")

        # --- Shape checks ---
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D with shape (n_features, n_samples); "
                f"got {X.ndim}D shape {X.shape}."
            )
        if Y.ndim != 2:
            raise ValueError(
                f"Y must be 2D with shape (n_output, n_samples); "
                f"got {Y.ndim}D shape {Y.shape}."
            )
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} samples (axis 1), Y has {Y.shape[1]}. They must match."
            )

        # --- Dimension checks against network's expected sizes ---
        expected_in  = self.n_neurons_each_layer[0]
        expected_out = self.layers[-1].n_neurons_output
        if X.shape[0] != expected_in:
            raise ValueError(
                f"X has {X.shape[0]} features along axis 0, "
                f"but the network expects {expected_in}."
            )
        if Y.shape[0] != expected_out:
            raise ValueError(
                f"Y has {Y.shape[0]} elements along axis 0, "
                f"but the network expects {expected_out}."
            )

        # --- Forward pass + per-element loss + mean ---
        Y_pred = self.prediction(X)
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
        X_train, Y_train : np.ndarray
            Training data in DATA-WORLD layout (rows = samples).
            X_train shape: (n_samples, n_features)
            Y_train shape: (n_samples, n_output) or (n_samples,) for single-output targets
        X_val, Y_val     : same layout, validation set
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
        X_train = np.asarray(X_train, dtype=float)
        Y_train = np.asarray(Y_train, dtype=float)
        X_val   = np.asarray(X_val,   dtype=float)
        Y_val   = np.asarray(Y_val,   dtype=float)

        # --- Promote 1D label arrays to 2D so .T behaves correctly ---
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)
        if Y_val.ndim == 1:
            Y_val = Y_val.reshape(-1, 1)

        # --- Cross the boundary: data-world (rows = samples) → math-world (columns = samples) ---
        X_train = X_train.T          # (n_features, n_samples)
        Y_train = Y_train.T          # (n_output,   n_samples)
        X_val   = X_val.T
        Y_val   = Y_val.T

        n_samples = X_train.shape[1]    # samples are along axis 1 now

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

            # Shuffle along the SAMPLE axis (axis 1, not axis 0)
            indices = self.rng.permutation(n_samples)
            X_shuffled = X_train[:, indices]        # reorder columns
            Y_shuffled = Y_train[:, indices]

            # Mini-batch loop — slice columns
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_X = X_shuffled[:, start:end]  # (n_features, B)
                batch_Y = Y_shuffled[:, start:end]  # (n_output,   B)
                self.compute_gradients_batch(batch_X, batch_Y)
                for layer in self.layers:
                    layer.update_parameters(current_lr, l2_lambda)

            # Track losses for this epoch (full train + val pass, both math-world)
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

    def save_model(self, output_filename, data_name, path="../models"):
        """Save model parameters in a consistent and replicable way."""

        if not isinstance(output_filename, str):
            raise TypeError("output_filename must be a string")
        if not isinstance(data_name, str):
            raise TypeError("data_name must be a string")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        full_path = save_dir / output_filename

        with open(full_path, "w", encoding="utf-8") as file:
            # --- Metadata ---
            file.write(f">Model: {output_filename}\n")
            file.write(f">Data used to train: {data_name}\n")
            file.write(f">Number of samples: {self.n_samples}\n")
            file.write(f">Epochs: {self.epochs}\n")
            file.write(f">Learning rate: {self.learning_rate}\n")
            file.write(f">Batch size: {self.batch_size}\n")
            file.write(f">Learning rate decay: {self.lr_decay}\n")
            file.write(f">Decay every: {self.decay_every}\n")
            file.write(f">L2 lambda: {self.l2_lambda}\n")

            # --- Architecture ---
            arch_str = ",".join(str(n) for n in self.n_neurons_each_layer)
            file.write(f">N layers: {self.n_layers}\n")
            file.write(f">Architecture: {arch_str}\n")
            file.write(f">Activation hidden: {self.activation_hidden}\n")
            file.write(f">Activation output: {self.activation_output}\n")
            file.write(f">Loss function: {self.loss_function}\n")

            # --- Parameters per layer ---
            for i, layer in enumerate(self.layers):
                n_out, n_in = layer.weights.shape

                file.write(f">Layer {i} weights: {n_out}x{n_in}\n")
                for row in layer.weights:
                    file.write(" ".join(f"{w:.10f}" for w in row) + "\n")

                file.write(f">Layer {i} biases: {n_out}x1\n")
                for value in layer.biases[:, 0]:
                    file.write(f"{value:.10f}\n")

        print(f"Model saved to: {full_path.resolve()}")

    @classmethod
    def load_model(cls, filepath):
        """Reconstruct an ANN from a saved model file."""
        with open(filepath, "r", encoding="utf-8") as file:
            lines = [line.rstrip("\n") for line in file]

        # --- First pass: parse header lines into a dict ---
        headers = {}
        data_lines = []
        for line in lines:
            if line.startswith(">"):
                key, _, value = line[1:].partition(":")
                headers[key.strip()] = value.strip()
            data_lines.append(line)

        # --- Build the model from architecture info ---
        architecture = [int(n) for n in headers["Architecture"].split(",")]
        ann = cls(
            n_layers=int(headers["N layers"]),
            n_neurons_each_layer=architecture,
            activation_hidden=headers["Activation hidden"],
            activation_output=headers["Activation output"],
            loss_function=headers["Loss function"],
        )

        # --- Second pass: walk through lines and load weights/biases ---
        i = 0
        layer_idx = 0
        while i < len(data_lines):
            line = data_lines[i]

            if line.startswith(">Layer") and "weights" in line:
                # ">Layer 0 weights: 32x27"
                shape_str = line.split(":")[1].strip()
                n_out, n_in = (int(x) for x in shape_str.split("x"))

                # Read the next n_out lines as weight rows
                rows = []
                for j in range(n_out):
                    row = [float(v) for v in data_lines[i + 1 + j].split()]
                    rows.append(row)
                ann.layers[layer_idx].weights = np.array(rows)             # (n_out, n_in)
                i += 1 + n_out

            elif line.startswith(">Layer") and "biases" in line:
                shape_str = line.split(":")[1].strip()
                n_out, _ = (int(x) for x in shape_str.split("x"))

                values = [float(data_lines[i + 1 + j]) for j in range(n_out)]
                ann.layers[layer_idx].biases = np.array(values).reshape(-1, 1)   # (n_out, 1)
                i += 1 + n_out
                layer_idx += 1

            else:
                i += 1

        print(f"Model loaded from: {Path(filepath).resolve()}")
        return ann
    