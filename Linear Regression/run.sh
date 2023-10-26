#!/bin/bash

# This script runs main.ipynb for the Linear Regression.
# To run this script, navigate to the directory where the script is located and run the following command:
# chmod +x run.sh
# ./run.sh
# This will execute the notebook and export the results in a file called 'main.nbconvert.ipynb' in the current directory.


jupyter nbconvert --execute --to notebook main.ipynb
