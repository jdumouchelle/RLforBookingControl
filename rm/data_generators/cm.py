import os
import subprocess
import time
import math
import pickle
import copy
import numpy as np

import multiprocessing as mp

from typing import Dict, List, Tuple

from rm.envs.cm.utils import *


class DataGenerator(object):
    
    def __init__(self, env:"Environment", time_limit=300, n_procs:int=1):
        """
        Constructor for MLDataGenerator. 
        """
        self.env = env
        self.time_limit = time_limit
        self.n_procs = n_procs

        self.rng = np.random.RandomState()
    
        self.instance_num = 0
        self.train_data = []
        self.valid_data = []
        
        return
    

    def seed(self, seed:int):
        """
        Seeds the data generator for reproducability.  

        Params
        -------------------------
        seed:
            The seed in which to seed the random number generator.  
        """
        self.rng.seed(seed)
        return
    
        
    def generate_data(self, accept_probs:List[float], n_runs:int, valid:bool=False, inst_unique_fp:str=""):
        """
        Generates data for machine learning.

        Params
        -------------------------
        accept_probs:
            A list of the accept probabilities to consider the random acceptance over.
        n_runs: 
            The number of runs to obtain data for.
        valid:
            An indicator specificy if the set should be considered train of validation.  
        """
        s = time.time()
        
        print("Train:", not valid)
        print('Acceptance Probabilites:', accept_probs)
        print('# Runs per prob:', n_runs)
        
        print(f'Computing end-of-horizon observations.')
        procs_to_run = []
        for accept_prob in accept_probs:
            for run in range(n_runs):
                obs = self._simulate_booking_phase(accept_prob)
                procs_to_run.append(obs)

        print(f'Solving end-of-horizon with n_procs={self.n_procs}.')
        pool = mp.Pool(self.n_procs)
        results = []
        for i, obs in enumerate(procs_to_run):
            results.append(pool.apply_async(solve_cm_mp, (obs,)))
        results = [r.get() for r in results]

        # add set features.  done outside MP to avoid pytorch errors.
        for i in range(len(results)):
            results[i]["set_features"] = get_set_features(results[i]["obs"], self.env.num_periods)

        # store dataset
        if not valid:
            self.train_data = results
        else:
            self.valid_data = results

        print('Time to generate data:', time.time()-s)

        return
    

    def _simulate_booking_phase(self, accept_prob):
        """
        Simulates the booking phase for a given accept probability.  

        Params
        -------------------------
        accept_prob:
            The probabilty at which each request is accepted.  
        """
        obs = self.env.reset()

        while True:
            action = 0
            if len(obs.action_set) == 2 and self.rng.uniform() <= accept_prob:
                action = 1
            obs, reward, done = self.env.step(action)
            if done: 
                break

        return obs


def solve_cm_mp(obs):
    """ Solve end-of-horizon problems in parallel. """
    
    # get operational cost
    model = formulate_mip_gp(obs)
    model.optimize()

    operational_cost =  0
    if len(obs['requests']) > 0:
        operational_cost = - model.objVal

    # get features and label
    linear_features = get_linear_features(obs)

    # store results in dict
    res = {
        "operational_cost" : operational_cost,
        "revenue" : obs["revenue"],
        "expected_revenue" : obs["expected_revenue"],
        "linear_features" : linear_features,
        "obs" : obs,
    }

    return res
