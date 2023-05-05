#!/usr/bash
# for slurm usage, must compile mpi4py with openmpi that was compiled for slurm

module purge
module load anaconda3
conda activate hrcl
module load gcc/10.3.0-o57x6h
module load openmpi/4.1.4
export MPICC=$(which mpicc)
pip install mpi4py --no-cache-dir
