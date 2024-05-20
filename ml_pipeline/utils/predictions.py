"""
Module for generating predictions from machine learning models.

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import numpy as np

from keras.models import Sequential


def get_predictions(model, X_test):
    """
    Generate predictions for the provided test dataset using a specified model.

    This function has specific handling for neural network models, 
    where the output is rounded to produce binary labels.

    Parameters:
    - model (Model): The model to be used for generating predictions. This can be any model that has a 
    predict method, like instances of keras.models.Sequential.
    - X_test (array-like): Test dataset on which predictions are to be made.
    
    Returns:
    - array-like: An array of predictions. For neural network models, the predictions are rounded 
    to the nearest integer to represent class labels.
    """

    # Get the predictions
    y_pred = model.predict(X_test)

    # Neural network, rounding is needed
    if isinstance(model, Sequential):
        y_pred = np.round(y_pred)

    # Return the predictions
    return y_pred