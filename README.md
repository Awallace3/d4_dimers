# D4 dimers

## Objective
- The repository provides code for optimizing -D3 and -D4 parameters with
  python3 using C++ dispersion functions
- Helper functions can be found in the `src` directory for constructing pandas
  dataframes for with -D4 C6's for monomers and dimers along with plotting
  results.

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
### Plotting Code
- run `plot.py` to generate graphs used in the manuscript.
### Optimization Code 
- The `main.py` python script is not meant to be statically used, but rather
serve as a place to use the functions in the `src` directory.



## Contact
This project is still under development and not stable yet. If you want to use
this codebase and would like assistance, please contact me via email at
<a href="mailto:austinwallace196@gmail.com">austinwallace196@gmail.com</a>

