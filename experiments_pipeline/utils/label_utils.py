"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import pandas as pd
import numpy as np



# Flip the labels w.r.t the given percentages of red and white wines
def flip_labels(input_csv, flip_percentage_red, flip_percentage_white):
    # Load the DataFrame
    df = pd.read_csv(input_csv)
    
    # Ensure the percentages are between 0 and 1
    if not (0 <= flip_percentage_red <= 1) or not (0 <= flip_percentage_white <= 1):
        raise ValueError("[ERROR] Percentages must be between 0 and 1")
    
    # Separate the DataFrame into red and white wines
    red_wines = df[df['type'] == False]
    white_wines = df[df['type'] == True]
    
    # Calculate the number of labels to flip
    num_red_to_flip = int(len(red_wines) * flip_percentage_red)
    num_white_to_flip = int(len(white_wines) * flip_percentage_white)
    
    # Randomly select indices to flip
    red_indices_to_flip = np.random.choice(red_wines.index, num_red_to_flip, replace=False)
    white_indices_to_flip = np.random.choice(white_wines.index, num_white_to_flip, replace=False)
    
    # Flip the labels
    df.loc[red_indices_to_flip, 'type'] = True
    df.loc[white_indices_to_flip, 'type'] = False
    
    return df