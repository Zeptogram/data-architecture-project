"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import pandas as pd
import numpy as np

import random



# Add rows with generated features (and random label) to the training set and return the new set
# The percentage is a floating point number between 0.0 and 1.0
# The generation can be between a defined range (see the AddRowsDomain task in the pipeline) for each feature, by default it's "unrestricted" (in [-100, 100])
def add_rows(input_csv, percentage, ranges={}):
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



# Duplicate rows w.r.t the given wine types to consider and the percentage
# By default it flips the label but it also can use the same label in the duplicate rows
def duplicate_rows(input_csv, wine_types_to_consider, percentage, flip_label=True):
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