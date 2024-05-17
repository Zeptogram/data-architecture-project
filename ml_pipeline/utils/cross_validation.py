"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import numpy as np
import sklearn as sn

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

from keras.models import Sequential

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

import scipy.stats as st

from utils.predictions import get_predictions


# Metrics used by each fold during cross validation, needed for the SVM / Decision Tree function
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1_score': make_scorer(f1_score, average='macro')
}



def get_nn_scores(model, X, y):
    """
    TODO docstring

    Please note that this function fits the model on different data.
    """

    # This function only supports Neural Network
    if not isinstance(model, Sequential):
        return None

    # Lists to save the metrics of each fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Stratified 10-fold cross validation is used
    cv=StratifiedKFold(n_splits=10, shuffle=True)

    for train_index, test_index in cv.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = np.array(y)[train_index], np.array(y)[test_index]

        # Re-compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Training for each fold
        model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)
        
        # Get the predictions on test set of the current fold
        y_pred = get_predictions(model, X_test_fold)

        # Get the metrics of the current fold
        accuracy_fold = accuracy_score(y_test_fold, y_pred)
        precision_fold = precision_score(y_test_fold, y_pred)
        recall_fold = recall_score(y_test_fold, y_pred)
        f1_fold = f1_score(y_test_fold, y_pred)

        # Save the current metrics
        accuracy_scores.append(accuracy_fold)
        precision_scores.append(precision_fold)
        recall_scores.append(recall_fold)
        f1_scores.append(f1_fold)

    # Return the scores
    return accuracy_scores, precision_scores, recall_scores, f1_scores



def get_svm_dtc_scores(model, X, y):
    """
    TODO docstring
    """

    # This function only supports SVM and Decision Tree
    if not isinstance(model, svm.SVC) and not isinstance(model, DecisionTreeClassifier):
        return None
    
    # Stratified 10-fold cross validation is used
    cv=StratifiedKFold(n_splits=10, shuffle=True)

    # Perform the cross validation
    scores = sn.model_selection.cross_validate(model, X, y, cv=cv, scoring=scoring)

    # For each metric, retrieve the result of each fold
    accuracy_scores = scores['test_accuracy']
    precision_scores = scores['test_precision']
    recall_scores = scores['test_recall']
    f1_scores = scores['test_f1_score']

    # Return the scores
    return accuracy_scores, precision_scores, recall_scores, f1_scores