import os
import subprocess
import multiprocessing as mp
import time
import math
import pickle
import copy

import numpy as np

from rm.envs.vrp.product_weights import ProductWeights
from rm.envs.vrp.vrp_instance_generator import VRPInstanceGenerator
from rm.envs.vrp.booking_period_env import BookingPeriodEnv
from rm.envs.vrp.vrp_solver import VRPSolver
from rm.envs.vrp.observation import Observation
from rm.envs.vrp.environment import Environment

from rm.envs.vrp.utils import *

from typing import Dict, List, Tuple

class DataGenerator(object):
    
    
    def __init__(self, env:"Environment", data_dir_path:str, n_procs:int=1):
        """
        Constructor for MLDataGenerator. 

        Params
        -------------------------
        env:
            The initialized environment
        """
        self.env = env
        self.data_dir_path = data_dir_path
        self.n_procs = n_procs

        self.vrp = env.vrp
        self.solver = env.solver
        self.num_locations = env.num_locations
        self.num_products = env.num_products
        self.num_periods = env.num_periods

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
        
        # compute all end observations
        print(f'Computing end-of-horizon observations.')
        procs_to_run = []
        for accept_prob in accept_probs:
            for run in range(n_runs):
                end_state, instance_path = self._simulate_booking_phase(accept_prob, inst_unique_fp)
                procs_to_run.append((end_state, instance_path))

        # solve all end-of-horizon problems in parallel
        print(f'Solving end-of-horizon with n_procs={self.n_procs}.')
        pool = mp.Pool(self.n_procs)
        results = []
        for end_state, instance_path in procs_to_run:
            results.append(pool.apply_async(solve_vrp_mp, (self.env, end_state, instance_path)))
        results = [r.get() for r in results]

        if not valid:
            self.train_data = results
        else:
            self.valid_data = results

        print('Time to generate data:', time.time()-s)
        
        return
    
    def _simulate_booking_phase(self, accept_prob, inst_unique_fp):
        """
        Simulates the booking phase for a given accept probability.  

        Params
        -------------------------
        accept_prob:
            The probabilty at which each request is accepted.  
        """
        obs = self.env.reset()

        while True:

            if len(obs.action_set) == 1:
                u_ji = 0
            else:
                u_ji = self.rng.choice(obs.action_set, p=[1-accept_prob, accept_prob])

            obs, reward, done = self.env.step(u_ji)

            if done:
                break
            
        # compute operational cost for end state        
        end_state = obs.state_tuple
        inst_fp = inst_unique_fp.split("/")[-1]
        instance_path = self.data_dir_path + f'vrp/filo_data/instances/{inst_fp}_{self.instance_num}.vrp'

        self.instance_num += 1
        
        return end_state, instance_path

def solve_vrp_mp(env, end_state, instance_path):
    """ Solve end-of-horizon problems in parallel. """
    return env.compute_vrp_cost(end_state, instance_path)
