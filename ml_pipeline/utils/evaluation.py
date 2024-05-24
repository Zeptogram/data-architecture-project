"""
Module for evaluating model performance on the wine dataset.

This module contains functions to calculate global performance metrics and confidence intervals for different 
machine learning models (Neural Networks, Support Vector Machines, and Decision Trees). 

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import scipy.stats as st

from keras.models import Sequential

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from utils.predictions import get_predictions

from utils.cross_validation import get_nn_scores, get_svm_dtc_scores



def get_global_metrics(model, X_test, y_test):
    """
    Compute basic performance metrics for a model based on the test dataset.

    This function evaluates a model's performance by predicting the test dataset and then calculating 
    accuracy, precision, recall, and F1-score. 

    Parameters:
    - model (Model): The trained model to evaluate, which must implement a predict method.
    - X_test (array-like): The feature set of the test data.
    - y_test (array-like): The actual labels for the test data.

    Returns:
    - dict: A dictionary containing the calculated metrics ('accuracy', 'precision', 'recall', 'f1_score').
    """

    # Retrieve the predictions
    y_pred = get_predictions(model, X_test)

    # Get the global metrics
    global_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Return the global metrics
    return global_metrics



def get_confidence_intervals(model, X, y):
    """
    Calculate 95% confidence intervals for the model's performance metrics using Cross Validation.

    It supports Neural Networks, SVMs, and Decision Trees, thanks to specific utility functions designed for these models. 

    Parameters:
    - model (Model): The model to evaluate. Must be either a Neural Network, SVM, or Decision Tree.
    - X (array-like): The full feature dataset used for Cross Validation.
    - y (array-like): The corresponding labels for the dataset.

    Returns:
    - dict: A dictionary with confidence intervals for each metric 
    ('accuracy_interval', 'precision_interval', 'recall_interval', 'f1_score_interval') if the model is supported. 
    Returns None if the model is unsupported.
    """

    # Neural Network
    if isinstance(model, Sequential):
        accuracy_scores, precision_scores, recall_scores, f1_scores = get_nn_scores(model, X, y)
    
    # SVM / Decision Tree
    elif isinstance(model, svm.SVC) or isinstance(model, DecisionTreeClassifier):
        accuracy_scores, precision_scores, recall_scores, f1_scores = get_svm_dtc_scores(model, X, y)
    
    # Model is not supported
    else:
        return None
    
    # Calculate the 95% confidence intervals
    accuracy_interval = st.t.interval(confidence=0.95, df=len(accuracy_scores)-1, loc=np.mean(accuracy_scores), scale=st.sem(accuracy_scores))
    precision_interval = st.t.interval(confidence=0.95, df=len(precision_scores)-1, loc=np.mean(precision_scores), scale=st.sem(precision_scores))
    recall_interval = st.t.interval(confidence=0.95, df=len(recall_scores)-1, loc=np.mean(recall_scores), scale=st.sem(recall_scores))
    f1_score_interval = st.t.interval(confidence=0.95, df=len(f1_scores)-1, loc=np.mean(f1_scores), scale=st.sem(f1_scores))

    # Dictionary containing the intervals
    intervals = {
        'accuracy_interval': accuracy_interval,
        'precision_interval': precision_interval,
        'recall_interval': recall_interval,
        'f1_score_interval': f1_score_interval
    }

    # Return the intervals
    return intervals