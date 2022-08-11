#!/usr/bin/env python


def make_config(
    num_sampling_strategies,
    parameters,
    objectives,
    seed,
    **kwargs,
    ):
    ''' generate Gryffin config given parameters and objectives

    Args:
        num_sampling_strategies (int): the number of sampling_strategies used,
            also the batch size of the experiments
        parameters: list of dictionaries containing the parameters of the experiment
        objectives: the objective(s) of the experiments
    '''
    config = {
        'general': {
            "num_cpus": 4,
            "auto_desc_gen": False,
            "batches": 1,
            "sampling_strategies": num_sampling_strategies,
            "boosted":  False,
            "caching": True,
            "random_seed": seed,
            "acquisition_optimizer": "genetic",
            "verbosity": 3
        },
        'parameters': parameters,
        'objectives': objectives,

    }

    return config
