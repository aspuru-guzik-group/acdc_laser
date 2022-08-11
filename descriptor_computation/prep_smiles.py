#!/usr/bin/env python

import os, sys
import numpy as np
import pandas as pd

a_frags = pd.read_csv('a_frags.csv', index_col=None)

b_frags = pd.read_csv('b_frags.csv', index_col=None)

c_frags = pd.read_csv('c_frags.csv', index_col=None)


for ix, row in a_frags.iterrows():
    row = row.to_dict()
    os.makedirs(f'fragments/{row["hid"]}', exist_ok=False)
    with open(f'fragments/{row["hid"]}/s.smi', 'w') as f:
        f.write(row['smiles_w_sub']+'\n')

for ix, row in b_frags.iterrows():
    row = row.to_dict()
    os.makedirs(f'fragments/{row["hid"]}', exist_ok=False)
    with open(f'fragments/{row["hid"]}/s.smi', 'w') as f:
        f.write(row['smiles_w_sub']+'\n')

for ix, row in c_frags.iterrows():
    row = row.to_dict()
    os.makedirs(f'fragments/{row["hid"]}', exist_ok=False)
    with open(f'fragments/{row["hid"]}/s.smi', 'w') as f:
        f.write(row['smiles_w_sub']+'\n')
