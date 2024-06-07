import os
import math
import time

import numpy as np
import gurobipy as gp

from typing import Dict, List, Tuple 

from .utils import *
from .observation import Observation
from .environment import Environment
from .rl_environment import GymEnvironment


class PolicyEvaluator(object):

    def __init__(self, env:"Environment", n_runs:int, seed:int, time_limit:int = 300, print_stats:bool=True, print_freq:int=100):
        """
        Constructor for PolicyEvaluator.

        Params
        -------------------------
        env:
            An initialized Environment.
        n_runs:
            the number of runs to evaluate over.
        seed:
            the seed which will be hold constant across different model.  
        """
        self.env = env
        self.n_runs = n_runs
        self.seed = seed

        self.time_limit = time_limit

        self.print_stats = print_stats
        self.print_freq = print_freq

        self.env.set_reproducible_costs(True)
        

    def evaluate_control(self, control_policy:"", use_gym=False, is_dqn=False, state_type="set", use_threshold=False):
        """
        Evalutes control policy model.  

        Params
        -------------------------
        control_policy:
            An instance of acontrol_policy model which has been trained.  
            Can be one of SarsaLinear, SarsaNN, MC, or any policy which
            contains the method get action.  
        """
        start_time = time.time()


        policy_profits = []

        for run in range(self.n_runs):

            self.env.seed(self.seed + run)

            if ((run+1) % self.print_freq) == 0 and self.print_stats:
                print(f"  Run: {run+1}/{self.n_runs} ")

            profit = self._simulate_control_policy(control_policy, self.env, use_gym, is_dqn, state_type, use_threshold)
            policy_profits.append(profit)


        end_time = time.time()

        profit_mean = np.mean(policy_profits)
        profit_std = np.std(policy_profits)
        results = {
            'profits' : policy_profits,
            'profit_mean' : profit_mean,
            'profit_std' : profit_std,
            'total_time' : end_time - start_time,
            'average_time' : (end_time - start_time)/self.n_runs,
        }

        if self.print_stats:
            print(f"Time: {results['total_time']}")
            print(f"Time (average): {results['average_time']}")

        return results

    def _simulate_control_policy(self, control_policy:"", env:"Environment", use_gym, is_dqn, state_type, use_threshold):
        """
        Simulates control policy.  

        Params
        -------------------------
        control_policy:
            An instance of acontrol_policy model which has been trained.  
            Can be one of SarsaLinear, SarsaNN, MC, or any policy which
            contains the method get action.  
        env:
            An initialized Environment.  

        """
        if use_gym:
            try:
                use_oh_period = control_policy.use_oh_period
            except:
                use_oh_period = 1 # True by default
            env_gym = GymEnvironment(
                        env = env,
                        ml_model = None, 
                        n_final_obs = 1, 
                        use_means = False, 
                        reproducible_costs = True,
                        use_oh_period = use_oh_period,
                        state_type = state_type)
            obs_gym = env_gym.reset()
                    
        obs = env.reset()
        
        revenue = 0
        t = 0
        while True:
            if use_gym:
                if is_dqn:
                    action = control_policy.compute_action(obs_gym, explore=False)
                else:
                    action = control_policy.get_action(obs_gym)

                # override with threshold
                if self.at_threshold(env, obs):
                    action = 0
                    
                if len(obs.action_set) == 1:
                    action = 0
                obs, reward, done = env.step(action)
                obs_gym, reward_gym, done_gym, _ = env_gym.step(action)
                
            else:
                action = control_policy.get_action(obs)
                obs, reward, done = env.step(action)
                
            if done:
                final_obs = obs
                break

        profit = self._compute_profit(final_obs)

        return profit
 

    def _compute_profit(self, final_obs):
        """
        """
        # get operational cost
        model = formulate_mip_gp(final_obs)
        model.setParam('limits/time', self.time_limit)
        model.optimize()

        operational_cost =  0
        if len(final_obs['requests']) > 0:
            operational_cost = - model.objVal

        profit = final_obs["revenue"] + operational_cost

        return profit


    def at_threshold(self, env, obs):
        def get_utilized_weight_volume(env):
            item_count = np.zeros(env.n_items)
            for key, value in env.state.items():
                item_count[key[0]] += value
            used_weight = np.array(env.weights) @ np.array(item_count)
            used_volume = np.array(env.volumes) @ np.array(item_count)
            return used_weight, used_volume

        used_weight, used_volume = get_utilized_weight_volume(env)
        if env.weights[obs.request[0]] + used_weight > env.expected_weight_capacity:
            return True
        elif env.volumes[obs.request[0]] + used_volume > env.expected_volume_capacity:
            return True
        return False