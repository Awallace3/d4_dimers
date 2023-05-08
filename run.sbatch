#!/bin/bash
#SBATCH -JSAPT0_adz                                # Job name
#SBATCH -Ahive-cs207                            # Charge account
#SBATCH -N8 -n100                            # Number of nodes and GPUs required
#SBATCH --mem=200G  
#SBATCH -t24:00:00                                        # Duration of the job (Ex: 15 mins)
#SBATCH -phive                                  # Partition name (where job is submitted)
#SBATCH -prog-o%j.out                             # Combined output and error messages file
#SBATCH --mail-type=END,FAIL                  # Mail preferences
#SBATCH --mail-user=awallace3@gatech.edu            # e-mail address for notifications

cd $HOME/data/gits/d4_dimers
module purge
module load anaconda3
conda activate hrcl
module load gcc/10.3.0-o57x6h
module load openmpi/4.1.4
conda activate hrcl
RUN_COMMAND=${RUN_COMMAND:-srun --mpi=openmpi}

srun python3 db.py
