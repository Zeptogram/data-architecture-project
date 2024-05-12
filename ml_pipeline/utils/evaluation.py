"""
TODO docstring

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

from predictions import get_predictions

from cross_validation import get_nn_scores, get_svm_dtc_scores



def get_global_metrics(model, X_test, y_test):
    """
    TODO docstring
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
    TODO docstring
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
        
