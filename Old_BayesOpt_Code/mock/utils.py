#!/usr/bin/env python


import os
import numpy as np
import pandas as pd


#---------------------------
# MOCK EXPERIMENT FUNCTIONS
#---------------------------

def run_experiment(param):
    ''' Run the mock experiment (fluorescence peak score
    as the objective)

    Args:
        param (dict): dictionary containing the parameters
    '''
    a_ix = int(param['A'][5:])
    b_ix = int(param['B'][5:])
    c_ix = int(param['C'][5:])
    match = df_results.loc[(df_results.frag_a_ix == a_ix) &
                           (df_results.frag_b_ix == b_ix) &
                           (df_results.frag_c_ix == c_ix)]
    assert len(match) == 1
    peak_score = match.loc[:, 'fluo_peak_1'].to_numpy()[0]
#    overlap = match.loc[:, 'overlap'].to_numpy()[0]
#    fluo_rate = match.loc[:, 'fluo_rate_ns'].to_numpy()[0]

    return peak_score


def eval_merit(param):
    ''' Run the mock experiment and update the parameter
    dictionary

    Args:
        param (dict): dictionary containing the parameters
    '''
    peak_score, overlap, fluo_rate = run_experiment(param)
    param['peak_score'] = peak_score

    return param


def query_database(
        df_results,
        observations,
        blacklist_a,
        blacklist_b,
        blacklist_c,
    ):
    ''' Divide the full database into smaller dataframes corresponding to
    the observed molecules and the candidates that a currently "makeable", i.e.
    that are not "in_progress" or

    observations = [ {"frag_a": frag_a_ix, "frag_b": frag_b_ix, "frag_c": frag_c_ix}, ..., ]


    Returns:
        df_makeable (pd.DataFrame): all the makeable molecules at a given iteration. This includes the
            molecules which have already been made, those that have not been made, and those that are in progress

    '''
    df_makeable = df_results[
        ~(df_results.frag_a_ix.isin(blacklist_a))&
        ~(df_results.frag_b_ix.isin(blacklist_b))&
        ~(df_results.frag_c_ix.isin(blacklist_c))
    ]

    cols = df_makeable.columns
    df_inprog = pd.DataFrame({col: [] for col in cols})

    return df_makeable, df_inprog


class MolQueue(object):
    ''' Molecular candidate queue object. Wraps a dictionary in its data attribute
    '''

    def __init__(self, max_num=None):

        self.max_num = max_num
        self.is_init = False

    def initialize(self, samples):
        self.data = samples
        for d in self.data:
            d['votes']=1

        self.is_init = True

    def update(self, samples):
        return None

    def remove(self):
        return None

    def to_disk(self):
        return None

    def from_disk(self):
        return None
