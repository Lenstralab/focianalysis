# FociAnalysis
Lenstra lab code for foci analysis as used by Meeussen et al. (2022).

## Cloning the repository
Install git: https://git-scm.com/

    git clone https://github.com/Lenstralab/focianalysis.git
    cd focianalysis

## Installation
If not done already:
- Install python (at least 3.8): https://www.python.org
- Install pip and git

Then install the focianalysis script (up to 5 minutes):

    pip install numpy cython pythran packaging ipython
    pip install -e .[smfish]

This will install the focianalysis package in your personal path in editable mode.

## Running
### Analysis pipeline
    foci_pipeline foci_parameter_file.yml

### Plotting
    foci_figures foci_figures_parameter_file.yml

## Testing
This script was tested with python 3.10 on Ubuntu 20.04 and on Mac OSX.
