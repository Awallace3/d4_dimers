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
conda env create -f env.yml
```
2. (Optional) The DFTD4 binary must be compiled from [this forked
   repository](https://github.com/Awallace3/dftd4) to acquire C6's and pass the
   pytests.

## Usage
### Plotting Code
- data is stored in `./plots/basis_study.pkl`; however, the file size is quite large 
    and fragmented into smaller files for storing in this repository. 
- After creating a python environment above, run `python3 plot.py` to generate
  `./plots/basis_study.pkl` and plot graphs shown in the manuscript.
### Optimization Code 
- The `main.py` python script is not meant to be statically used, but rather
serve as a place to use the functions in the `src` directory.

## Contact
If you want to use this codebase and would like assistance, please contact me
via email at <a href="mailto:austinwallace196@gmail.com">austinwallace196@gmail.com</a>
