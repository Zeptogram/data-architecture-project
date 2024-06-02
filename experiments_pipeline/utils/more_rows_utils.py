"""
This module provides functions to augment a dataset by adding or duplicating rows with generated features and labels.

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import pandas as pd
import numpy as np

import random



def add_rows(input_csv, percentage, ranges={}):
    """
    Add rows with generated features (and random label) to the training set and return the new set.
    
    Parameters:
    - input_csv (str): Path to the input csv file.
    - percentage (float): Percentage of rows to add, as a floating point number between 0.0 and 1.0.
    - ranges (dict): Optional. A dictionary specifying the range for each feature to generate the values. 
    If not specified, the default range is [-100, 100] for each feature.
    
    Returns:
    - pd.DataFrame: A new DataFrame with the additional generated rows.
    
    Raises:
    - ValueError: If the percentage is not between 0 and 1.
    """

    # Ensure the percentage is between 0 and 1
    if not (0 <= percentage <= 1):
        raise ValueError("[ERROR] Percentage must be between 0 and 1")

    # Load the DataFrame
    df = pd.read_csv(input_csv)

    # If percentage is 0, do nothing
    if percentage == 0:
        return df

    # Dict containing all the rows to add
    rows_to_add_dict = {}
    # Fill the dict with empty lists and also the missing ranges in the argument
    for column in df.columns:
        # Dynamically set the columns to support dropping features
        rows_to_add_dict[column] = [] 
        # If the range of a feature is not set, use [-100, 100]
        if column != 'type' and not (column in ranges):
            ranges[column] = (-100, 100)

    # Calculate how many rows to add
    n_rows_to_add = int(len(df) * percentage)

    # Generate the rows
    for i in range(0, n_rows_to_add):
        # Generate target and features
        for column in df.columns:
            # Generate the value (bool if target, else a float in the range)
            if column == 'type':
                generated_value = random.choice([True, False])
            else:
                generated_value = random.uniform(ranges[column][0], ranges[column][1])
            # Add the generated value
            rows_to_add_dict[column].append(generated_value)

    # Convert from dict to DataFrame
    rows_to_add = pd.DataFrame(rows_to_add_dict)

    # Append the generated rows
    df = pd.concat([df, rows_to_add], ignore_index=True)

    # Return the new training set
    return df



def duplicate_rows(input_csv, wine_types_to_consider, percentage, flip_label=True):
    """
    Duplicate rows based on the given wine types to consider and the percentage.
    By default, it flips the label but it can also use the same label in the duplicate rows.
    
    Parameters:
    - input_csv (str): Path to the input csv file.
    - wine_types_to_consider (list or tuple): List or tuple of wine types to consider for duplication. Can contain 'red' or 'white' or both.
    - percentage (float): Percentage of rows to duplicate, as a floating point number between 0.0 and 1.0.
    - flip_label (bool): Optional. Whether to flip the label of the duplicated rows. Default is True.
    
    Returns:
    - pd.DataFrame: A new DataFrame with the duplicated rows.
    
    Raises:
    - ValueError: If the percentage is not between 0 and 1 or if the wine types are not 'red' or 'white'.
    """

    # Convert tuple to list if necessary
    if isinstance(wine_types_to_consider, tuple):
        wine_types_to_consider = list(wine_types_to_consider)

    # Ensure the percentage is between 0 and 1
    if not (0 <= percentage <= 1):
        raise ValueError("[ERROR] Percentage must be between 0 and 1")

    # Load the DataFrame
    df = pd.read_csv(input_csv)

    # If percentage is 0, do nothing
    if percentage == 0:
        return df
    
    # Check that the passed wine types are valid
    for type in wine_types_to_consider:
        if not type in ['red', 'white']:
            raise ValueError("[ERROR] Wine types can only be 'red' or 'white'")

    # Prepare a list to collect the duplicated rows
    rows_to_duplicate = []
    
    # Handle duplication for each wine type
    for wine_type in wine_types_to_consider:
        if wine_type == 'red':
            selected_rows = df[df['type'] == False]
        elif wine_type == 'white':
            selected_rows = df[df['type'] == True]
        
        # Calculate the number of rows to duplicate
        num_rows_to_duplicate = int(len(selected_rows) * percentage)
        
        # Randomly select rows to duplicate
        duplicated_indices = np.random.choice(selected_rows.index, num_rows_to_duplicate, replace=False)
        
        # Append the selected rows to the list
        rows_to_duplicate.append(df.loc[duplicated_indices])
    
    # Concatenate all the duplicated rows into a single DataFrame
    duplicated_df = pd.concat(rows_to_duplicate)
    
    # Optionally flip the labels
    if flip_label:
        duplicated_df['type'] = ~duplicated_df['type']
    
    # Append the duplicated rows to the original DataFrame
    df = pd.concat([df, duplicated_df], ignore_index=True)

    return df