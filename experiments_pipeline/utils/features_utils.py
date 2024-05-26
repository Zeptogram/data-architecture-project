"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import pandas as pd



# Drop the features from the training set and return the new set
def drop_features(train_csv, features):
    # ListParameter creates a tuple instead of a list, convert it before using
    if isinstance(features, tuple):
        features = list(features)
    
    # Load the DataFrame
    df = pd.read_csv(train_csv)

    # Drop the features
    df.drop(columns=features, inplace=True)

    # Return the new training set
    return df