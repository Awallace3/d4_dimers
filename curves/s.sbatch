#!/bin/bash
#SBATCH -JSAPTDFT
#SBATCH -oSAPTDFT.out
#SBATCH --open-mode=append
#SBATCH -Ahive-cs207
#SBATCH -N10 --ntasks=10 --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH -t72:00:00
#SBATCH -phive
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu

cd $HOME/data/sapt_dft_ddftd4/code_ddft_d4/curves/
. "/storage/home/hhive1/awallace43/data/miniconda3/etc/profile.d/conda.sh"
conda activate /storage/hive/project/chem-sherrill/awallace43/.conda/envs/p4dev19
echo "CONDA_PREFIX=" ${CONDA_PREFIX}

export PATH=/storage/hive/project/chem-sherrill/awallace43/gits/psi4_amw/build_delta_dft/stage/bin:$PATH 
export PYTHONPATH=/storage/hive/project/chem-sherrill/awallace43/gits/psi4_amw/build_delta_dft/stage/lib:$PYTHONPATH
export SCRATCH=${TMPDIR}
export PSI_SCRATCH=${TMPDIR}

echo "setting scratch to: ${TMPDIR}"

echo "starting job..."
mem_per_cpu_gb=$((SLURM_MEM_PER_CPU / 1024))
mem_per_process=$((mem_per_cpu_gb * SLURM_CPUS_ON_NODE))

n_procs=$SLURM_NTASKS

echo "mem_per_cpu_gb = " $mem_per_cpu_gb
echo "SLURM_CPUS_ON_NODE = " $SLURM_CPUS_ON_NODE
echo "mem_per_process = " $mem_per_process
echo "n_procs = " $n_procs

mpiexec -n $n_procs python3 -u ./jobs.py
