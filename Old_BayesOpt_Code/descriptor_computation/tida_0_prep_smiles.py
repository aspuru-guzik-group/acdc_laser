#!/usr/bin/env python

import os, sys
import numpy as np
import pandas as pd

new_tidas = pd.read_csv('BTIDA_Min.csv', index_col=None)


for ix, row in new_tidas.iterrows():
    row = row.to_dict()
    os.makedirs(f'tidas/{row["hid"]}', exist_ok=False)
    with open(f'tidas/{row["hid"]}/s.smi', 'w') as f:
        f.write(row['smiles_w_sub']+'\n')
