#!/bin/bash
IDS=('031' '032' '033' '034' '035' '036' '041' '043' '050' '051' '052' '053' '054' '055' '057')
for id in "${IDS[@]}"
do
	echo "Working on $id"
	bash gen_confs_crest.sh ./tidas/B"$id" s.smi 10 mmff94
done
