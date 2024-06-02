"""
This module generates a JSON configuration file for various experiments by defining different sets of parameters.

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import json

# Define the data
data = []

def main():
    """
    Generate a JSON configuration file for various experiments.
    
    The function defines different sets of parameters for experiments, including:
    - Drop features experiments
    - Missing values experiments
    - Outliers experiments
    - Out of domain values experiments
    - Flip labels experiments
    - Duplicate rows with the same label experiments
    - Duplicate rows with the opposite label experiments
    - Add rows with random values experiments
    - Add rows within a domain experiments
    - Combination of multiple modifications
    
    The generated configuration is saved to 'experiments_config.json'.
    """
    
    # Generate drop features single experiments
    exp_number = 1
    to_consider = [["PC1"], ["PC2"], ["PC1", "PC2"], ["PC3", "PC4", "PC5"], ["PC1", "PC2", "PC3", "PC4"]]
    for features_to_drop in to_consider:
        data.append(
            {
                "experiment-name": f"experiment #{exp_number}",
                "features-to-drop": features_to_drop,
            }
        )
        exp_number += 1

    # Pop last element
    to_consider.pop()

    # Generate missing values single experiments
    percentages = [x / 100 for x in range(10, 100, 10)]
    classes = [["red"], ["white"], ["red", "white"]]
    for percentage in percentages:
        for class_list in classes:
            for features_to_consider in to_consider:
                data.append(
                    {
                        "experiment-name": f"experiment #{exp_number}",
                        "features-to-dirty-mv": features_to_consider,
                        "missing-values-percentage": percentage,
                        "wine-types-to-consider-missing-values": class_list
                    }
                )
                exp_number += 1

    # Append 100% percentage
    percentages.append(1.0)

    # Generate outliers single experiments
    ranges = ["std", "iqr"]
    for percentage in percentages:
        for class_list in classes:
            for range_type in ranges:
                for features_to_consider in to_consider:
                    data.append(
                        {
                            "experiment-name": f"experiment #{exp_number}",
                            "features-to-dirty-outliers": features_to_consider,
                            "outliers-percentage": percentage,
                            "wine-types-to-consider-outliers": class_list,
                            "range-type": range_type
                        }
                    )
                    exp_number += 1

    # Generate out of domain single experiments
    for percentage in percentages:
        for class_list in classes:
            for features_to_consider in to_consider:
                data.append(
                    {
                        "experiment-name": f"experiment #{exp_number}",
                        "features-to-dirty-oodv": features_to_consider,
                        "oodv-percentage": percentage,
                        "wine-types-to-consider-oodv": class_list,
                    }
                )
                exp_number += 1

    # Flip labels single experiments
    for percentage_red in percentages:
        for percentage_white in percentages:
            data.append(
                {
                    "experiment-name": f"experiment #{exp_number}",
                    "flip-percentage-red": percentage_red,
                    "flip-percentage-white": percentage_white
                }
            )
            exp_number += 1

    # Duplicate rows same label single experiments
    for percentage in percentages:
        for class_list in classes:
            data.append(
                {
                    "experiment-name": f"experiment #{exp_number}",
                    "wine-types-to-consider-same-label": class_list,
                    "duplicate-rows-same-label-percentage": percentage
                }
            )
            exp_number += 1

    # Duplicate rows opposite label single experiments
    for percentage in percentages:
        for class_list in classes:
            data.append(
                {
                    "experiment-name": f"experiment #{exp_number}",
                    "wine-types-to-consider-opposite-label": class_list,
                    "duplicate-rows-opposite-label-percentage": percentage
                }
            )
            exp_number += 1

    # Add rows random single experiments
    for percentage in percentages:
        data.append(
            {
                "experiment-name": f"experiment #{exp_number}",
                "add-rows-random-percentage": percentage
            }
        )
        exp_number += 1

    # Add rows domain single experiments
    for percentage in percentages:
        data.append(
            {
                "experiment-name": f"experiment #{exp_number}",
                "add-rows-domain-percentage": percentage
            }
        )
        exp_number += 1

    # Add combos
    
    data.append(
        {
            "experiment-name": f"experiment #{exp_number}",
            "features-to-drop": ["PC3", "PC4", "PC5"],
            "features-to-dirty-mv": ["PC1", "PC2"],
            "missing-values-percentage": 0.15,
            "features-to-dirty-outliers": ["PC1"],
            "outliers-percentage": 0.05,
            "features-to-dirty-oodv": ["PC2"],
            "oodv-percentage": 0.05,
            "flip-percentage-red": 0.03,
            "flip-percentage-white": 0.03,
            "duplicate-rows-same-label-percentage": 0.1,
            "duplicate-rows-opposite-label-percentage": 0.05,
            "add-rows-random-percentage": 0.015,
            "add-rows-domain-percentage": 0.03
        }
    )
    exp_number += 1

    data.append(
        {
            "experiment-name": f"experiment #{exp_number}",
            "features-to-drop": ["PC1", "PC2"],
            "features-to-dirty-mv": ["PC3", "PC4", "PC5"],
            "missing-values-percentage": 0.5,
            "flip-percentage-red": 0.1,
            "flip-percentage-white": 0.45,
            "add-rows-random-percentage": 0.35
        }
    )
    exp_number += 1

    data.append(
        {
            "experiment-name": f"experiment #{exp_number}",
            "features-to-drop": ["PC1", "PC2", "PC3", "PC4"],
            "features-to-dirty-outliers": ["PC5"],
            "outliers-percentage": 0.02,
        }
    )
    exp_number += 1

    data.append(
        {
            "experiment-name": f"experiment #{exp_number}",
            "features-to-dirty-mv": ["PC1", "PC2", "PC3", "PC4", "PC5"],
            "missing-values-percentage": 0.3,
            "flip-percentage-red": 0.1,
            "flip-percentage-white": 0.15
        }
    )
    exp_number += 1

    data.append(
        {
            "experiment-name": f"experiment #{exp_number}",
            "features-to-drop": ["PC1"],
            "flip-percentage-red": 1.0,
            "flip-percentage-white": 1.0,
            "add-rows-random-percentage": 0.2
        }
    )
    exp_number += 1

    data.append(
        {
            "experiment-name": f"experiment #{exp_number}",
            "features-to-dirty-mv": ["PC2", "PC3"],
            "missing-values-percentage": 0.35,
            "features-to-dirty-oodv": ["PC1", "PC4", "PC5"],
            "oodv-percentage": 0.5,
            "add-rows-domain-percentage": 0.7
        }
    )
    exp_number += 1

    # Convert the data to a JSON string
    json_data = json.dumps(data, indent=4)

    # Save the JSON string to a file
    with open('experiments_config.json', 'w') as json_file:
        json_file.write(json_data)

    print(json_data)


if __name__ == '__main__':
    main()