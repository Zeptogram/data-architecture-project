"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import numpy as np

from keras.models import Sequential


def get_predictions(model, X_test):
    """
    TODO docstring
    """

    # Get the predictions
    y_pred = model.predict(X_test)

    # Neural network, rounding is needed
    if isinstance(model, Sequential):
        y_pred = np.round(y_pred)

    # Return the predictions
    return y_pred