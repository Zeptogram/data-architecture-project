"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import pandas as pd

import sys

import random



# Add rows with generated features (and random label) to the training set and return the new set
# The percentage is a floating point number between 0.0 and 1.0
# The generation can be between a defined range (see the AddRowsDomain task in the pipeline) for each feature, by default it's "unrestricted" (in [-1e10, 1e10])
def add_rows(input_csv, percentage, ranges={}):
    # Load the DataFrame
    df = pd.read_csv(input_csv)

    # Dict containing all the rows to add
    rows_to_add_dict = {}
    # Fill the dict with empty lists and also the missing ranges in the argument
    for column in df.columns:
        # Dynamically set the columns to support dropping features
        rows_to_add_dict[column] = [] 
        # If the range of a feature is not set, use [-1e10, 1e10]
        if column != 'type' and not (column in ranges):
            ranges[column] = (-1e10, 1e10)

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