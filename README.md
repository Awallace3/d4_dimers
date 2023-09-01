# D4 dimers

## Objective
- The repository provides code for optimizing -D3 and -D4 parameters with python3
- Helper functions can be found in the `src` directory for constructing pandas
  dataframes for with -D4 C6's for monomers and dimers along with plotting
  results.
- Currently, two forms of five-fold cross validation optimization procedures
  are implemented: Powell Algorithm and Levenberg-Marquardt least-squares
  fitting.  

[ ] - add d4/jdz, ATM w/ charged C6's to ATM plot.
[ ] - plot SR function results.
[ ] - Investigate other damping functions.
[ ] - add MBD code here to use for RMSE violin plots

```
export OMP_NUM_THREADS=8
```

## Installation
1. Recommended installation is to create a conda environment using the
   `environment.yml` file provided.
```bash
conda env create -f environment.yml
```
2. The DFTD4 binary must be compiled from [this forked
   repository](https://github.com/Awallace3/dftd4) to acquire C6's and pass the
   pytests.

## Usage
The `main.py` python script is not meant to be statically used, but rather
serve as a place to use the functions in the `src` directory.


## Contact
This project is still under development and not stable yet. If you want to use
this codebase and would like assistance, please contact me via email at
<a href="mailto:austinwallace196@gmail.com">austinwallace196@gmail.com</a>

# Plan
[ ] - Plot violin results for ATM and basis set individually.
[ ] - Plot close contacts and explain what is happening.
[ ] - Determine parameter sensitivity.
