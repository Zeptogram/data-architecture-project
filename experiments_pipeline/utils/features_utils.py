"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import pandas as pd
import numpy as np


# Drop the features from the training set and return the new set
def drop_features(input_csv, features_to_drop):
    # ListParameter creates a tuple instead of a list, convert it before using
    if isinstance(features_to_drop, tuple):
        features_to_drop = list(features_to_drop)
    
    # Load the DataFrame
    df = pd.read_csv(input_csv)

    # Drop the features
    df.drop(columns=features_to_drop, inplace=True)

    # Return the new training set
    return df


def introduce_missing_values(input_csv, features_to_dirty, percentage):
    # Convert tuple to list if necessary
    if isinstance(features_to_dirty, tuple):
        features_to_dirty = list(features_to_dirty)
    
    # Load the DataFrame
    df = pd.read_csv(input_csv)
    
    # Ensure the percentage is between 0 and 1
    if not (0 <= percentage <= 1):
        raise ValueError("[ERROR] Percentage must be between 0 and 1")
    
    # Introduce missing values
    for feature in features_to_dirty:
        if feature in df.columns:
            # Calculate the number of values to replace with NaN
            num_values = int(percentage * len(df))
            # Randomly select indices to replace with NaN
            indices = np.random.choice(df.index, num_values, replace=False)
            df.loc[indices, feature] = np.nan
        else:
            raise ValueError(f"[ERROR] Feature '{feature}' not found in the DataFrame")
    
    # Return the new training set
    return df