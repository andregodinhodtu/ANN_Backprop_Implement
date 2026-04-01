import random
import numpy as np

class ANN():

    ACTIVATION_FUNCTION = {
        # sigmoid for binary classification problem
        "sigmoid": {
            "func": lambda x: 1.0 / (1.0 + np.exp(-x)),
            "deriv": lambda a: a * (1.0 - a)
        }
    }

    LOSS_FUNCTION = {
        # uses the raw output before sigmoid
        # output z -> probability sigma(z)


        # binary cross entropy for binary classification problem
        "bce": {
            "func": lambda y, y_pred: -np.mean(
                y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
            ),
            "deriv": lambda y, y_pred: (
                -y / y_pred + (1 - y) / (1 - y_pred)
            ) / y.size
            
                                               
        }
    }


    