"""
This module provides functions to manipulate features in the dataset, such as dropping features, introducing missing values, outliers, and out of domain values.

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import pandas as pd
import numpy as np


dirty_level = 10 # For OOD, it's the constant that multiplies std



def drop_features(input_csv, features_to_drop):
    """
    Drop the features from the training set and return the new set.
    
    Parameters:
    - input_csv (str): Path to the input csv file.
    - features_to_drop (list or tuple): List or tuple of features to drop from the dataset.
    
    Returns:
    - pd.DataFrame: A new DataFrame with the specified features dropped.
    
    Raises:
    - ValueError: If an attempt is made to drop the target or a feature not found in the DataFrame.
    """

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



def introduce_missing_values(input_csv, wine_types_to_consider, features_to_dirty, percentage):
    """
    Introduce missing values in the given set.
    
    Parameters:
    - input_csv (str): Path to the input csv file.
    - wine_types_to_consider (list or tuple): List or tuple of wine types to consider ('red' or 'white' or both).
    - features_to_dirty (list or tuple): List or tuple of features to introduce missing values into.
    - percentage (float): Percentage of values to replace with NaN, as a floating point number between 0.0 and 1.0.
    
    Returns:
    - pd.DataFrame: A new DataFrame with the specified percentage of missing values introduced.
    
    Raises:
    - ValueError: If the percentage is not between 0 and 1, or if an attempt is made to introduce missing values to the target,
    or if the wine types are not 'red' or 'white'.
    """

    # Convert tuple to list if necessary
    if isinstance(features_to_dirty, tuple):
        features_to_dirty = list(features_to_dirty)
    # Convert tuple to list if necessary
    if isinstance(wine_types_to_consider, tuple):
        wine_types_to_consider = list(wine_types_to_consider)
    # Load the DataFrame
    df = pd.read_csv(input_csv)
    # Ensure the percentage is between 0 and 1
    if not (0 <= percentage <= 1):
        raise ValueError("[ERROR] Percentage must be between 0 and 1")
    # If percentage is 0, do nothing
    if percentage == 0:
        return df
    # Error 'type' can't be a features_to_dirty
    if 'type' in features_to_dirty:
        raise ValueError("[ERROR] Can't affect the target with missing values")

    # Check that the passed wine types are valid
    for type in wine_types_to_consider:
        if type not in ['red', 'white']:
            raise ValueError("[ERROR] Wine types can only be 'red' or 'white'")
    
    # Select the rows based on wine types
    if len(wine_types_to_consider) == 2:
        selected_rows = df
    else:
        if wine_types_to_consider[0] == 'red':
            # Red = False
            selected_rows = df[df['type'] == False]
        elif wine_types_to_consider[0] == 'white':
            # White = True
            selected_rows = df[df['type'] == True]
    # Introduce missing values
    for feature in features_to_dirty:
        if feature in df.columns:
            # Calculate the number of values to replace with NaN
            num_values = int(percentage * len(selected_rows))
            # Randomly select indices to replace with NaN
            indices = np.random.choice(selected_rows.index, num_values, replace=False)
            df.loc[indices, feature] = np.nan
        else:
            raise ValueError(f"[ERROR] Feature '{feature}' not found in the DataFrame")  
    # Return the modified DataFrame
    return df



def introduce_outliers(input_csv, df_ranges, wine_types_to_consider, features_to_dirty, percentage, range_type="std"):
    """
    Introduce outliers in the given set, using std + mean or IQR.
    
    Parameters:
    - input_csv (str): Path to the input csv file.
    - df_ranges (pd.DataFrame): DataFrame containing the original training set, used to get ranges for the features.
    - wine_types_to_consider (list or tuple): List or tuple of wine types to consider ('red' or 'white' or both).
    - features_to_dirty (list or tuple): List or tuple of features to introduce outliers into.
    - percentage (float): Percentage of values to replace with outliers, as a floating point number between 0.0 and 1.0.
    - range_type (str): Type of range to use for generating outliers. Can be 'std' or 'iqr'. Default is 'std'.
    
    Returns:
    - pd.DataFrame: A new DataFrame with the specified percentage of outliers introduced.
    
    Raises:
    - ValueError: If the percentage is not between 0 and 1, or if an attempt is made to introduce outliers to the target,
    or if the wine types are not 'red' or 'white'.
    """

    # Convert tuple to list if necessary
    if isinstance(features_to_dirty, tuple):
        features_to_dirty = list(features_to_dirty)
    # Load the DataFrame
    df = pd.read_csv(input_csv)
    # Ensure the percentage is between 0 and 1
    if not (0 <= percentage <= 1):
        raise ValueError("[ERROR] Percentage must be between 0 and 1")
    # If percentage is 0, do nothing
    if percentage == 0:
        return df
    # Error 'type' can't be a features_to_dirty
    if 'type' in features_to_dirty:
        raise ValueError("[ERROR] Can't affect the target with outliers values")
    # Check that the passed wine types are valid
    for type in wine_types_to_consider:
        if type not in ['red', 'white']:
            raise ValueError("[ERROR] Wine types can only be 'red' or 'white'")


    # Get the ranges for the features
    ranges = get_ranges(df_ranges, features_to_dirty, range_type)

    # Select the rows based on wine types
    if len(wine_types_to_consider) == 2:
        selected_rows = df
    else:
        if wine_types_to_consider[0] == 'red':
            # Red = False
            selected_rows = df[df['type'] == False]
        elif wine_types_to_consider[0] == 'white':
            # White = True
            selected_rows = df[df['type'] == True]
    
    # Introduce outliers
    for feature in features_to_dirty:
        if feature in df.columns:
            # Drop NaN values for the feature
            non_nan_indices = selected_rows[feature].dropna().index
            # Calculate the number of outliers to introduce
            num_values = int(percentage * len(non_nan_indices))
            # Randomly select indices to replace with outliers
            indices = np.random.choice(non_nan_indices, num_values, replace=False)
            # Get the ranges for the feature
            lower_bound, upper_bound = ranges[feature]
            # Get min and max value
            min_value, max_value = df[feature].min(), df[feature].max()
            # Assign outliers randomly as either high or low
            for idx in indices:
                if np.random.rand() > 0.5:
                    # Generate high outlier
                    outlier_value = np.random.uniform(upper_bound, max_value)
                else:
                    # Generate low outlier
                    outlier_value = np.random.uniform(min_value, lower_bound)
                # Ensure the outlier is within min and max limits of the feature
                outlier_value = max(min(outlier_value, max_value), min_value)
                df.at[idx, feature] = outlier_value
        else:
            raise ValueError(f"[ERROR] Feature '{feature}' not found in the DataFrame")
    # Return the new training set
    return df



def introduce_oodv(input_csv, wine_types_to_consider, features_to_dirty, percentage):
    """
    Introduce out of domain values.
    
    Parameters:
    - input_csv (str): Path to the input csv file.
    - wine_types_to_consider (list or tuple): List or tuple of wine types to consider ('red' or 'white' or both).
    - features_to_dirty (list or tuple): List or tuple of features to introduce out of domain values into.
    - percentage (float): Percentage of values to replace with out of domain values, as a floating point number between 0.0 and 1.0.
    
    Returns:
    - pd.DataFrame: A new DataFrame with the specified percentage of out of domain values introduced.
    
    Raises:
    - ValueError: If the percentage is not between 0 and 1, or if an attempt is made to introduce out of domain values to the target,
    or if the wine types are not 'red' or 'white'.
    """

    # Convert tuple to list if necessary
    if isinstance(features_to_dirty, tuple):
        features_to_dirty = list(features_to_dirty)
    # Load the DataFrame
    df = pd.read_csv(input_csv)
    # Ensure the percentage is between 0 and 1
    if not (0 <= percentage <= 1):
        raise ValueError("[ERROR] Percentage must be between 0 and 1")
    # If percentage is 0, do nothing
    if percentage == 0:
        return df
    # Error 'type' can't be a features_to_dirty
    if 'type' in features_to_dirty:
        raise ValueError("[ERROR] Can't affect the target with out of domain values")
    # Check that the passed wine types are valid
    for type in wine_types_to_consider:
        if type not in ['red', 'white']:
            raise ValueError("[ERROR] Wine types can only be 'red' or 'white'")
    # Select the rows based on wine types
    if len(wine_types_to_consider) == 2:
        selected_rows = df
    else:
        if wine_types_to_consider[0] == 'red':
            # Red = False
            selected_rows = df[df['type'] == False]
        elif wine_types_to_consider[0] == 'white':
            # White = True
            selected_rows = df[df['type'] == True]
    # Introduce out of domain values
    for feature in features_to_dirty:
        if feature in df.columns:
             # Drop NaN values for the feature
            non_nan_indices = selected_rows[feature].dropna().index
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



def get_ranges(df, features, range_type = "std", threshold_std = 3, threshold_iqr = 2):
    """
    Using std mean or IQR, get the ranges for the features.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the original training set.
    - features (list or tuple): List or tuple of features to calculate the ranges for.
    - range_type (str): Type of range to use for calculation. Can be 'std' or 'iqr'. Default is 'std'.
    - threshold_std (int): Threshold for std deviation. Default is 3.
    - threshold_iqr (int): Threshold for IQR. Default is 2.
    
    Returns:
    - dict: A dictionary with features as keys and their calculated ranges as values.
    """

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