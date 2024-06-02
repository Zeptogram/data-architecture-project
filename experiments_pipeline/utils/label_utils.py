"""
This module provides functions to manipulate labels in the dataset, i.e. flipping labels for a given percentage of red and white wines.

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import pandas as pd
import numpy as np



def flip_labels(input_csv, flip_percentage_red, flip_percentage_white):
    """
    Flip the labels with respect to the given percentages of red and white wines.
    
    Parameters:
    - input_csv (str): Path to the input csv file.
    - flip_percentage_red (float): Percentage of red wine labels to flip, as a floating point number between 0.0 and 1.0.
    - flip_percentage_white (float): Percentage of white wine labels to flip, as a floating point number between 0.0 and 1.0.
    
    Returns:
    - pd.DataFrame: A new DataFrame with the flipped labels.
    
    Raises:
    - ValueError: If the flip percentages are not between 0 and 1.
    """

    # Load the DataFrame
    df = pd.read_csv(input_csv)
    
    # Ensure the percentages are between 0 and 1
    if not (0 <= flip_percentage_red <= 1) or not (0 <= flip_percentage_white <= 1):
        raise ValueError("[ERROR] Percentages must be between 0 and 1")
    
    # If percentages are both 0, do nothing
    if flip_percentage_red == 0 and flip_percentage_white == 0:
        return df
    
    # Separate the DataFrame into red and white wines
    red_wines = df[df['type'] == False]
    white_wines = df[df['type'] == True]
    
    # Calculate the number of labels to flip
    num_red_to_flip = int(len(red_wines) * flip_percentage_red)
    num_white_to_flip = int(len(white_wines) * flip_percentage_white)

    # Randomly select indices to flip, and flip the labels
    
    if num_red_to_flip > 0:
        red_indices_to_flip = np.random.choice(red_wines.index, num_red_to_flip, replace=False)
        df.loc[red_indices_to_flip, 'type'] = True

    if num_white_to_flip > 0:
        white_indices_to_flip = np.random.choice(white_wines.index, num_white_to_flip, replace=False)
        df.loc[white_indices_to_flip, 'type'] = False
    
    return df