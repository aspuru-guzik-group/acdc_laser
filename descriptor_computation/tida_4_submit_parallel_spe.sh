#!/bin/bash
#SBATCH -A rrg-aspuru
#SBATCH --nodes 4
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 40
#SBATCH --time=23:59:59
#SBATCH --job-name desc_spe
#SBATCH --output=job_%j

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$HOME/xtb/bin:$PATH

# pythonmo
module load CCEnv
module load StdEnv
module load gnu-parallel
# module load intelpython3 gnu-parallel
# conda activate madness-spectra

export QCHEMPATH=/scinet/niagara/software/commercial/qc54
export QCAUX=$QCHEMPATH/qcaux
export QCSCRATCH=$(pwd)
. $QCHEMPATH/qcenv.sh

parallel qchem {1}spe_input.inp {1}spe.out :::: ./tida_spe_joblist.txt