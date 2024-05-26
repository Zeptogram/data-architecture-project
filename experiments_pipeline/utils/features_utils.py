"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import pandas as pd



# Drop the features from the training set and return the new set
def drop_features(train_csv, features_to_drop):
    # ListParameter creates a tuple instead of a list, convert it before using
    if isinstance(features_to_drop, tuple):
        features_to_drop = list(features_to_drop)
    
    # Load the DataFrame
    df = pd.read_csv(train_csv)

    # Drop the features
    df.drop(columns=features_to_drop, inplace=True)

    # Return the new training set
    return df