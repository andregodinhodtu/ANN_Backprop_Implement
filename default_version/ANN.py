##################################
# Build Aritificial Neural Network
##################################
import random
from ANN_layer import ANN_Layer
import math

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
    def log(cls, x):
        return math.log(x)
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
                
                # Computed activation derivatives for the layer
                layer.compute_activation_derivatives()
                
                for j in range(layer.n_neurons_output):
                    a = layer.a_s[j][0]           # activation
                    target = y[j][0]              # true value
                    delta = self.LOSS_FUNCTIONS[self.loss_function]["deriv"](a, target) * layer.activation_derivatives[j]
                    layer.delta.append(delta)
            else:
                # Hidden layers
                next_layer = self.layers[i + 1]
                layer.delta = []
                
                # Computed activation derivatives for the layer
                layer.compute_activation_derivatives()
                
                for j in range(layer.n_neurons_output):
                    # Sum over next layer's deltas weighted by corresponding weights
                    weighted_sum = sum(
                        next_layer.delta[k] * next_layer.weights[k][j]
                        for k in range(next_layer.n_neurons_output)
                    )
                    delta = weighted_sum * layer.activation_derivatives[j]
                    layer.delta.append(delta)

    def compute_gradients_sample(self, input_vector, target):
        """
        Computes gradients (dweights and dbiases) for a single training sample.

        input_vector: input column vector (list of lists)

        target: target output column vector (list of lists)
        """
        n_layers = len(self.layers)

        # Forward pass (uses your existing method)
        self.prediction(input_vector)

        # Compute deltas
        self._compute_deltas(target)
        
        # Compute gradients
        for i in range(n_layers):
            layer = self.layers[i]

            # Determine input to this layer
            if i == 0:
                prev_activations = input_vector
            else:
                prev_activations = self.layers[i - 1].a_s

            # Initialize gradients
            layer.dweights = [
                [0.0 for _ in range(len(prev_activations))]
                for _ in range(layer.n_neurons_output)
            ]
            layer.dbiases = [0.0 for _ in range(layer.n_neurons_output)]

            # Compute gradients
            for j in range(layer.n_neurons_output):
                for k in range(len(prev_activations)):
                    layer.dweights[j][k] = layer.delta[j] * prev_activations[k][0]

                layer.dbiases[j] = layer.delta[j]
    
    def compute_gradients_batch(self, batch_inputs, batch_targets):
        batch_size = len(batch_inputs)

        # Initialize accumulators using n_neurons_input (safe at this point)
        accum_dweights = []
        accum_dbiases = []
        for layer in self.layers:
            accum_dweights.append([
                [0.0 for _ in range(layer.n_neurons_input)]
                for _ in range(layer.n_neurons_output)
            ])
            accum_dbiases.append([0.0 for _ in range(layer.n_neurons_output)])

        for x, y in zip(batch_inputs, batch_targets):
            self.compute_gradients_sample(x, y)
            for i, layer in enumerate(self.layers):
                for j in range(layer.n_neurons_output):
                    for k in range(len(layer.dweights[j])):
                        accum_dweights[i][j][k] += layer.dweights[j][k]
                    accum_dbiases[i][j] += layer.dbiases[j]

        for i, layer in enumerate(self.layers):
            for j in range(layer.n_neurons_output):
                for k in range(len(accum_dweights[i][j])):
                    accum_dweights[i][j][k] /= batch_size
                accum_dbiases[i][j] /= batch_size
            layer.dweights = accum_dweights[i]
            layer.dbiases = accum_dbiases[i]      
        
    def train(self, X, Y, epochs=10, learning_rate=0.01, batch_size=1, 
              verbose=True, lr_decay=0.95, decay_every=20):  # ← add these params
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
                    layer.update_parameters(current_lr)  # ← use current_lr

                for x_sample, y_sample in zip(batch_X, batch_Y):
                    pred = self.prediction(x_sample)
                    loss = self.LOSS_FUNCTIONS[self.loss_function]["func"](pred, y_sample)
                    epoch_loss += loss

            epoch_loss /= n_samples

            if verbose:
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}")
        