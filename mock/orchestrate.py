#!/usr/bin/env python

import os, sys
import time
import pickle

import numpy as np
import pandas as pd

from gryffin import Gryffin

from config import make_config
from utils import (
    run_experiment,
    query_database,
    MolQueue,

)

 # ..., get_params, get_objectives
# from db_comm import ...

"""
This script orchestrates a simulation of the MADNESS ACDC lasers project with
data reported in

"""

#---------------
# CONFIGURATION
#---------------

TIME_INTERVAL = 5             # time interval in seconds
NUM_SAMPLING_STRATEGIES = 40  # batch size
BUDGET = 2000                 # maximum number of measurements
RANDOM_SEED = 2022            # random seed for initial suggestions
SAMPLING_STRATEGIES = np.linspace(-1, 1, NUM_SAMPLING_STRATEGIES)

DUMP_PATH = 'dump'
PICKUP_PATH = 'pickup'

NUM_BLACKLIST_A = 4           # number of intially blacklisted fragments
NUM_BLACKLIST_B = 4
NUM_BLACKLIST_C = 5

BLACKLIST_REMOVE_PROB = 0.05  # per batch probability that a new fragment is added

prev_num_train = 0            # counter for current number of observations


#df_results = pd.read_csv('data/df_results.csv', index_col=0) # full dataset
df_results = pd.read_pickle('data/full_props.pkl') # full dataset
num_total_a = len(df_results.loc[:, 'frag_a_ix'].unique())
num_total_b = len(df_results.loc[:, 'frag_b_ix'].unique())
num_total_c = len(df_results.loc[:, 'frag_c_ix'].unique())

print(num_total_a, num_total_b, num_total_c)
print(df_results.shape)
print(df_results.head())

np.random.seed(RANDOM_SEED)

blacklist_init_a = np.random.choice(np.arange(num_total_a), size=(NUM_BLACKLIST_A), replace=False)
blacklist_init_b = np.random.choice(np.arange(num_total_b), size=(NUM_BLACKLIST_B), replace=False)
blacklist_init_c = np.random.choice(np.arange(num_total_c), size=(NUM_BLACKLIST_C), replace=False)

blacklist_a = blacklist_init_a
blacklist_b = blacklist_init_b
blacklist_c = blacklist_init_c


print('blacklist a : ', blacklist_init_a)
print('blacklist b : ', blacklist_init_b)
print('blacklist c : ', blacklist_init_c)





#----------------------
# BEGIN THE EXPERIMENT
#----------------------

mol_queue = MolQueue()

observations = []

while True:

    #------------------------------------
    # EXTRACT CURRENT DATA FROM DATABASE
    #------------------------------------


    df_makeable, df_inprog = query_database(
        df_results, observations, blacklist_a, blacklist_b, blacklist_c
    )
    print(df_makeable.shape, df_inprog.shape)

    # TODO: replace this with a check the observations
    if True:

        # get descriptors for all the fragments (None for naive Gryffin)
        A_frags = {f"frag_{i}":None for i in df_makeable.loc[: , 'frag_a_ix'].unique()}
        B_frags = {f"frag_{i}":None for i in df_makeable.loc[: , 'frag_b_ix'].unique()}
        C_frags = {f"frag_{i}":None for i in df_makeable.loc[: , 'frag_c_ix'].unique()}

        # parameters
        parameters = [
            {'name': 'frag_a', 'type': 'categorical', 'category_details': A_frags},
            {'name': 'frag_b', 'type': 'categorical', 'category_details': B_frags},
            {'name': 'frag_c', 'type': 'categorical', 'category_details': C_frags},
        ]

        # objectives
        objectives = [
            {'name': 'peak_score', 'goal': 'max'},
        ]



        gryffin_config = make_config(NUM_SAMPLING_STRATEGIES, parameters, objectives, RANDOM_SEED)

        def known_constraints(param):
            match_makeable = df_makeable.loc[
                (df_makeable.frag_a_ix==int(param['frag_a'][5:]))&
                (df_makeable.frag_b_ix==int(param['frag_b'][5:]))&
                (df_makeable.frag_c_ix==int(param['frag_c'][5:]))
            ]
            match_inprog  = df_inprog.loc[
                (df_inprog.frag_a_ix==int(param['frag_a'][5:]))&
                (df_inprog.frag_b_ix==int(param['frag_b'][5:]))&
                (df_inprog.frag_c_ix==int(param['frag_c'][5:]))
            ]
            if np.logical_and(
                len(match_makeable)==1, len(match_inprog)==0
            ):
                return True
            else:
                return False



        # TODO: add known constraints once we have a method of getting them
        gryffin = Gryffin(config_dict=gryffin_config, known_constraints=known_constraints)



        # recommend a batch of samples
        samples = gryffin.recommend(observations)

        # add sampling strategy to the samples
        for ix, sample in enumerate(samples):
            sample['sampling_strategy'] = SAMPLING_STRATEGIES[ix]

        if not mol_queue.is_init:
            # initialize the queue
            mol_queue.initialize(samples)
        else:
            # udpate the queue with new samples
            mol_queue.update(samples)
        print(mol_queue.data)









    time.sleep(TIME_INTERVAL)
