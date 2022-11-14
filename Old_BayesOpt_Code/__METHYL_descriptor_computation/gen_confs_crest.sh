#!/bin/bash

# generate conformers using openbabel and crest

args=("$@")

dir=${args[0]}
smiles=${args[1]}
nconfs=${args[2]}
ff=${args[3]}

cd $dir
obabel $smiles --gen3D --ff $ff -O step1.mol
obabel step1.mol --minimize --ff $ff -O step2.mol
obabel step2.mol --conformer --nconf $nconfs --ff $ff --writeconformers -O conformers_obabel.xyz
crest -screen conformers_obabel.xyz > opt.out
perl split_conformers crest_ensemble.xyz
