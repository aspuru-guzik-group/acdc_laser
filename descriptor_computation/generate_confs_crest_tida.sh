#!/bin/bash
IDS=('033' '034' '035' '036' '041' '043' '050' '051' '052' '054' '055' '057')
for id in "${IDS[@]}"
do
	echo "Working on $id"
	bash gen_confs_crest.sh ./fragments/B"$id" s.smi 10 mmff94
done
