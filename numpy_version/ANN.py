import random
import numpy as np

class ANN():

    """ANN algorithm with backpropagation made specifically for binary classification.
    This version utilizes NumPy for maximum efficiency and clarity. """



    # -----------------------------------------------------------------

    # sigmoid for binary classification problem
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(a):
        return a * (1.0 - a)
    

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
    


    # constructor
    def __init__(self, n_layers, n_neurons_each_layer): 

        # input check
        if not isinstance(n_layers, int):
            raise TypeError("n_layers must be an Integer")
        if not isinstance(n_neurons_each_layer, list):
            raise TypeError("n_neurons_each_layer should be a list")
        if not all(isinstance(n, int) for n in n_neurons_each_layer):
            raise TypeError("All elements in n_neurons_each_layer must be integers")
        
        # value check
        if n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if len(n_neurons_each_layer) != n_layers:
           raise ValueError("Length of n_neurons_each_layer must equal n_layers") 
        
        # init settings
        self.n_layers =  n_layers
        self.n_neurons_each_layer =  n_neurons_each_layer

        # init arrays
        self.weights = [
            # array shape of n_out, n_in for all layers except 1st
            # filled with random values between -0.5 and 0.5
            np.random.uniform(-0.5, 0.5, (n_out, n_in))
            for n_in, n_out in zip(n_neurons_each_layer[:-1], n_neurons_each_layer[1:])
        ]
        self.biases = [
            #for each layer after 1st the bias is set to 0.01
            np.full((n_out, 1), 0.01)
            for n_out in n_neurons_each_layer[1:]
        ]
    

    def predict(self, input_vector):

        # input as a column vector
        a = np.array(input_vector).reshape(-1, 1)
        
        # storage for pre-activation and activation values
        self.z_s = []
        self.a_s = [a] # node activation (initialized with input values)
        
        # forward pass
        for W, b in zip(self.weights, self.biases):
            # over weights and bias in each layer
            z = np.dot(W, a) + b    #a is the current activation value
            self.z_s.append(z)

            # update and save activation
            a = self.sigmoid(z)
            self.a_s.append(a)
            print(self.a_s)

        # output = last activated value
        return a



# test
ann = ANN(3, [5, 3, 1])
ann.predict([1, 2, 3, 4, 1])







    