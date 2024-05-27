"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import pandas as pd
import numpy as np


dirty_level = 10 # For OOD, it's the constant that multiplies std



# Drop the features from the training set and return the new set
def drop_features(input_csv, features_to_drop):
    # ListParameter creates a tuple instead of a list, convert it before using
    if isinstance(features_to_drop, tuple):
        features_to_drop = list(features_to_drop)
    # Load the DataFrame
    df = pd.read_csv(input_csv)
    # Make sure only valid features are passed (and not the target, which can't be dropped)
    for feature in features_to_drop:
        if feature == 'type':
            raise ValueError("[ERROR] Target can't be dropped")
        if not (feature in df.columns):
            raise ValueError(f"[ERROR] Feature '{feature}' not found in the DataFrame")  
    # Drop the features
    df.drop(columns=features_to_drop, inplace=True)
    # Return the new training set
    return df



# Introduces missing values in the given set
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



# Introduces outliers in the given set, using std + mean or IQR
def introduce_outliers(input_csv, features_to_dirty, percentage, range_type="std"):
    # Convert tuple to list if necessary
    if isinstance(features_to_dirty, tuple):
        features_to_dirty = list(features_to_dirty)
    # Load the DataFrame
    df = pd.read_csv(input_csv)
    # Ensure the percentage is between 0 and 1
    if not (0 <= percentage <= 1):
        raise ValueError("[ERROR] Percentage must be between 0 and 1")
    # Get the ranges for the features
    ranges = get_ranges(df, features_to_dirty, range_type)
    
    # Introduce outliers
    for feature in features_to_dirty:
        if feature in df.columns:
            # Drop NaN values for the feature
            non_nan_indices = df[feature].dropna().index
            # Calculate the number of outliers to introduce
            num_values = int(percentage * len(non_nan_indices))
            # Randomly select indices to replace with outliers
            indices = np.random.choice(non_nan_indices, num_values, replace=False)
            # Get the ranges for the feature
            lower_bound, upper_bound = ranges[feature]
            # Get min and max value
            min_value, max_value = df[feature].min(), df[feature].max()
            # Assign outliers randomly as either high or low with a small random variation
            for idx in indices:
                random_variation = np.random.uniform(-0.01, 0.01)
                if np.random.rand() > 0.5:
                    # Generate high outlier
                    outlier_value = np.random.uniform(upper_bound, max_value) + random_variation
                else:
                    # Generate low outlier
                    outlier_value = np.random.uniform(min_value, lower_bound) + random_variation
                
                # Ensure the outlier is within min and max limits of the feature
                outlier_value = max(min(outlier_value, max_value), min_value)
                df.at[idx, feature] = outlier_value
        else:
            raise ValueError(f"[ERROR] Feature '{feature}' not found in the DataFrame")
    # Return the new training set
    return df



# Introduces out of domain values
def introduce_oodv(input_csv, features_to_dirty, percentage):
    # Convert tuple to list if necessary
    if isinstance(features_to_dirty, tuple):
        features_to_dirty = list(features_to_dirty)
    # Load the DataFrame
    df = pd.read_csv(input_csv)
    # Ensure the percentage is between 0 and 1
    if not (0 <= percentage <= 1):
        raise ValueError("[ERROR] Percentage must be between 0 and 1")
    # Introduce out of domain values
    for feature in features_to_dirty:
        if feature in df.columns:
             # Drop NaN values for the feature
            non_nan_indices = df[feature].dropna().index
            # Calculate the number of out of domain values to introduce
            num_values = int(percentage * len(non_nan_indices))
            # Randomly select indices to replace with oodv
            indices = np.random.choice(non_nan_indices, num_values, replace=False)
             # Introduce OOD values by adding/subtracting a large constant (e.g., 10 times the standard deviation)
            ood_high = df[feature].mean() + dirty_level * df[feature].std()
            ood_low = df[feature].mean() - dirty_level * df[feature].std()
            # Assign OOD values randomly as either high or low
            for idx in indices:
                random_variation = np.random.uniform(-0.01, 0.01)
                df.at[idx, feature] = np.random.choice([ood_high, ood_low]) + random_variation
        else:
            raise ValueError(f"[ERROR] Feature '{feature}' not found in the DataFrame")
    # Return the new training set
    return df



# Using std mean or iqr, get the ranges
def get_ranges(df, features, range_type = "std", threshold_std = 5, threshold_iqr = 4):
    ranges = {}
    for feature in features:
        if range_type == "iqr":
            # Calculate the IQR
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold_iqr * IQR
            upper_bound = Q3 + threshold_iqr * IQR
        else:
            # Calculate the mean and std
            mean = df[feature].mean()
            std = df[feature].std()
            lower_bound = mean - threshold_std * std
            upper_bound = mean + threshold_std * std
        
        ranges[feature] = (lower_bound, upper_bound)
    return ranges
