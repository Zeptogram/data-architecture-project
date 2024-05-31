"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import json

import subprocess, os, sys



def main():
    # Load JSON experiments configuration
    with open("experiments_config.json", "r") as file:
        json_data = json.load(file)

    # Iterate through the JSON data
    for entry in json_data:
        # For each entry compute the parameters to pass to the Luigi pipeline, after getting the venv Python executable
        cmd_parameters = [os.path.join(sys.prefix, "Scripts", "python"), "-m", "luigi", "--module", "pipeline", "FitPerformanceEval"]
        for key, value in entry.items():
            # Append the key
            cmd_parameters.append(f"--{key}")
            # Handle the type correctly
            if isinstance(value, list):
                value = json.dumps(value)
            else:
                value = str(value)
            # Append the value
            cmd_parameters.append(value)

        # Append --local-scheduler
        cmd_parameters.append("--local-scheduler")

        print(f"Running experiment with parameters: {cmd_parameters}")
        
        # Execute the pipeline with the parameters
        subprocess.run(cmd_parameters, check=True)
    


if __name__ == '__main__':
    main()