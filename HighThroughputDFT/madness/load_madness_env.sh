#!/bin/bash
module --force purge
module load gcc/12.2.0  openmpi/4.1.4 
module load python/3.10.8
module load rdkit/2022.03.5
module load orca/5.0.3
module load openbabel/3.1.1
source $HOME/madness/madness_env/bin/activate

export PATH=$PATH:$HOME/xtb/bin
export XTBPATH=$HOME/xtb/share/xtb
export KMP_NUM_THREADS=$1
export MKL_NUM_THREADS=$1
export KMP_STACKSIZE="1000M"
ulimit -s unlimited

export Multiwfnpath=$HOME/Multiwfn_3.8
export PATH=$PATH:$HOME/Multiwfn_3.8

export SPECTRA=$HOME/bin/spectra
export PATH=$PATH:$SPECTRA:$SPECTRA/bin
