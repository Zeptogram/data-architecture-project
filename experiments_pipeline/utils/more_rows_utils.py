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
# The generation can be between a defined range (Min-Max), by default it's in [-1e10, 1e10] ("unrestricted")
def add_rows(input_csv, percentage, min=-1e10, max=1e10):
    # Load the DataFrame
    df = pd.read_csv(input_csv)

    # Dict containing all the rows to add
    rows_to_add_dict = {}
    for column in df.columns:
        rows_to_add_dict[column] = [] # dynamically set the columns to support dropping features

    # Calculate how many rows to add
    n_rows_to_add = int(len(df) * percentage)

    # Generate the rows
    for i in range(0, n_rows_to_add):
        # Generate target and features
        for column in df.columns:
            # Generate the value (bool if target, else a float in [min, max])
            if column == 'type':
                generated_value = random.choice([True, False])
            else:
                generated_value = random.uniform(min, max)
            # Add the generated value
            rows_to_add_dict[column].append(generated_value)

    # Convert from dict to DataFrame
    rows_to_add = pd.DataFrame(rows_to_add_dict)

    # Append the generated rows
    df = pd.concat([df, rows_to_add], ignore_index=True)

    # Return the new training set
    return df