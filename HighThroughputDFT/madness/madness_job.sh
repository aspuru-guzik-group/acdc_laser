#!/bin/bash
#SBATCH --account=aspuru
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00-3:00
#SBATCH --signal=B:15@30
#SBATCH --ntasks=ppppp
#SBATCH --nodes=1
##SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=clascclon@gmail.com
#SBATCH --job-name=nnnnn

np=ppppp
nt=$(( np * 2 ))
charge=0
spin=1

smi="smismismi"
name="nnnnn"
input="inputinputinput"
# Define variables
dir=$SLURM_SUBMIT_DIR
scriptdir=$HOME/bin/projects/spectra

function cleanup_submitnext() {
  deactivate
  cp $dir/sl*.out $workdir/
  cp -rf $workdir/* $dir
  rm -rf $workdir

  cd $dir
  cd ..
  tar -cf "/project/madness/all_hid/"$name".tar" ./$name && rm -rf ./$name 
  bash submit.sh $input

  exit 0
}
trap cleanup_submitnext SIGTERM EXIT

echo "job start"
echo $(date +%T)

#load environment
source $HOME/madness/load_madness_env.sh $nt

# Create folders and files
cp -rf $HOME/madness/jobtemplate/* $dir
#workdir=$SLURM_TMPDIR/$name
workdir=$TMPDIR/$name
mkdir $workdir
cp -rf $dir/* $workdir

# Use cheapocrest to get lowest conformer
cd $workdir/crest
python cheapocrest-python.py --nconfs 50 --ff uff --chrg $charge --theory gfnff --solvent acetonitrile $smi >cheapocrest.out
cp ./crest_best.xyz ../opt

# Get number of atoms in molecule
natoms=$(cat ./crest_best.xyz | head -n1)
nxyz=$(( natoms + 2 ))
#head -n $nxyz ./crest_conformers.xyz > ./crest_best.xyz

echo "crest done"
echo $(date +%T)

# Perform opt job
cd ../opt
$xtbdir/xtb crest_best.xyz --chrg $charge --uhf $((spin-1)) --opt verytight --gfn 2 -P $nt > ./opt.out

# ReOrient xyz file and send to ../hess folder 
python reorientxyz.py 

# Perform hess job, generate hess.npy
cd ../hess
$xtbdir/xtb reorient.xyz --chrg $charge --uhf $((spin-1)) --hess --gfn 2 -P $nt > ./hess.out
python xtbhess2np.py $natoms
cp ./reorient.xyz ../orcajobs/S0_opt.xyz
cp ./reorient.xyz $SCRATCH/all_hid/processed_xyz/$name.xyz

echo "xtb done"
echo $(date +%T)

# Perform UV-VIS absorption spectrum prediction
cd ../orcajobs/

# Run ORCA spectra calculation
/software/avx2/software/ORCA/5.0.3-gompi-2022b/bin/orca S0_tdsp.inp  | tee -a S0_tdsp.out $dir/orcajobs/S0_tdsp.out

#transform the standard orca .gbw wavefunction output to molden format for Multiwfn to read
/software/avx2/software/ORCA/5.0.3-gompi-2022b/bin/orca_2mkl S0_tdsp -molden
mv S0_tdsp.molden.input S0_tdsp.molden

nele=$(awk '/Number of Electrons/{print $NF }' S0_tdsp.out)
echo $nele >> ../spec/orca_mo.out
grep "ORBITAL ENERGIES" S0_tdsp.out -A$((nele + 3)) >> ../spec/orca_mo.out
grep "CARTESIAN GRADIENT" ./S0_tdsp.out -A$((natoms+2)) >> grads.out

echo "orca done"
echo $(date +%T)

Multiwfn ./S0_tdsp.molden -nt $np < ./ES_info.txt | tee ./ES_info.out
#Multiwfn ./S0_tdsp.gbw -nt $np < ./ES_info.txt | tee ./ES_info.out

grep "Sr index" ./ES_info.out |nl >> ES_results.out; echo >> ES_results.out
grep "D index" ./ES_info.out |nl >> ES_results.out; echo >> ES_results.out
grep "RMSD of hole in" ./ES_info.out |nl >> ES_results.out; echo >> ES_results.out
grep "RMSD of electron in" ./ES_info.out |nl >> ES_results.out; echo >> ES_results.out
grep "H index" ./ES_info.out |nl >> ES_results.out; echo >> ES_results.out
grep " t index" ./ES_info.out |nl >> ES_results.out; echo >> ES_results.out
grep "Delta_r =" ./ES_info.out |nl >> ES_results.out; echo >> ES_results.out
grep "lambda =" ./ES_info.out |nl >> ES_results.out; echo >> ES_results.out
grep "Transition electric dipole moment between ground state" ./ES_info.out -A7 >> ES_results.out
grep "Transition electric dipole moment between excited states" ./ES_info.out -A17 >> ES_results.out

echo "Multiwfn done"
echo $(date +%T)

python engrad2np.py $natoms
python mu2np.py

cd ../spec
cp ../hess/hessian.npy .
cp ../orcajobs/dipmoments.npy .
cp ../orcajobs/grads.npy .
cp ../orcajobs/excenergies.npy .
cp ../orcajobs/S0_opt.xyz .
cp ../orcajobs/ES_results.out .

# Caldulate spectra, seems there is bug?
python $HOME/bin/spectra/spectra_vg_corr.py -grid all
echo "all done"
echo $(date +%T)

# Extract descriptors
cd ..
python Extract_ORCA_Descriptors.py
cp *.json $SCRATCH/all_hid/processed_json

wait

