import os
import time
import numpy as np
import gurobipy as gp

from typing import Dict, List, Tuple 

from .utils import *
from .environment import Environment
from .rl_environment import GymEnvironment


class PolicyEvaluator(object):

    def __init__(self, data_dir:str, env:"Environment", n_runs:int, seed:int, print_stats:bool=True, print_freq:int=10):
        """
        Constructor for PolicyEvaluator.

        Params
        -------------------------
        env:
            An initialized environment.
        n_runs:
            the number of runs to evaluate over.
        seed:
            the seed which will be hold constant across different model.  
        """
        self.data_dir = data_dir
        self.env = env
        self.n_runs = n_runs
        self.seed = seed

        self.print_stats = print_stats
        self.print_freq = print_freq


    def evaluate_dp(self, dp_model, use_exact:bool=True):
        """
        Evalutes dynamic programming model.  

        Params
        -------------------------
        dp_model:
            An instance of ExactDP which has already computed the value function.
        use_exact:
            An indicator to determine if the exact policy vrp or ml vrp costs should be used.  
        """
        start_time = time.time()

        self.env.seed(self.seed)
        policy_profits = []


        for run in range(self.n_runs):

            if ((run+1) % self.print_freq) == 0 and self.print_stats:
                print(f"  Run: {run+1}/{self.n_runs} ")

            profit = self._simulate_value_function_policy(dp_model, self.env, use_exact)
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
            print(f"Mean profit: {results['profit_mean']}")
            print(f"Time: {results['total_time']}")
            print(f"Time (average): {results['average_time']}")

        return results


    def evaluate_control(self, control_policy, use_gym, is_dqn, state_type, use_threshold=False):
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

        self.env.seed(self.seed)
        policy_profits = []


        for run in range(self.n_runs):
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
            print(f"Mean profit: {results['profit_mean']}")
            print(f"Time: {results['total_time']}")
            print(f"Time (average): {results['average_time']}")

        return results

    def evaluate_baseline(self, baseline_policy):
        """
        Evalutes baseline policy.

        Params
        -------------------------
        blp:
            An instance of GGMBLP.
        """
        start_time = time.time()

        self.env.seed(self.seed)
        policy_profits = []

        for run in range(self.n_runs):

            if ((run+1) % self.print_freq) == 0 and self.print_stats:
                print(f"  Run: {run+1}/{self.n_runs} ")

            profit = baseline_policy.run(self.env)
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
            print(f"Mean profit: {results['profit_mean']}")
            print(f"Time: {results['total_time']}")
            print(f"Time (average): {results['average_time']}")

        return results

    
    def _simulate_value_function_policy(self, dp_model, env, use_exact:bool):
        """
        Simluates dynamic programming model.  

        Params
        -------------------------
        dp_model:
            An instance of ExactDP which has already computed the value function.
        env:
            An initialized env.  
        use_exact:
            An indicator to determine if the exact policy vrp or ml vrp costs should be used.  
        """
        if use_exact:
            value_function = dp_model.value_function_exact
        else:
            value_function = dp_model.value_function_ml

        obs = env.reset()
        revenue = 0
        t = 0
        while True:
            state = obs.state_tuple

            u_ji = 0
            if len(obs.action_set) > 1:
                next_state = dp_model.state_transition_dict[state][(obs.location, obs.product)]
                assert((np.array(next_state) - np.array(state)).sum() == 1)
                if env.prices[obs.location][obs.product] >= value_function[t+1][state] - value_function[t+1][next_state]:
                    u_ji = 1

            obs, reward, done = env.step(u_ji)

            revenue += reward
            t += 1

            if done:
                break

        # get demands at each location        
        end_state = obs.state_tuple
        instance_path = self.data_dir + 'vrp/filo_data/instances/simulation_end_state.vrp'

        vrp_results = env.compute_vrp_cost(end_state, instance_path)

        # compute profit
        profit = revenue - vrp_results['adjusted_operational_cost']

        return profit


    def _simulate_control_policy(self, control_policy, env, use_gym, is_dqn, state_type, use_threshold):
        """
        Simulates control policy.  

        Params
        -------------------------
        control_policy:
            An instance of acontrol_policy model which has been trained.  
            Can be one of SarsaLinear, SarsaNN, MC, or any policy which
            contains the method get action.  
        env:
            An initialized env.  
        """
        try:
            use_oh_period = control_policy.use_oh_period
        except:
            use_oh_period = 1 # True by default

        control_env = GymEnvironment(env, None, state_type=state_type, use_oh_period = use_oh_period)

        obs_gym = control_env.reset()
        obs = env.reset()
        
        total_reward = 0
        t = 0
        while True:
            if len(control_env.prev_obs.action_set) == 1:
                action = 0
            elif is_dqn:
                action = control_policy.compute_action(obs_gym, explore=False)
            elif use_gym:  
                action = control_policy.get_action(obs_gym)
            else:
                action = control_policy.get_action(obs)

            # override with threshold
            if self.at_threshold(env, obs):
                action = 0

            obs_gym, reward, done, _ = control_env.step(action)
            obs, reward_norm, _ = env.step(action)

            total_reward += reward
            if done:
                final_obs = obs
                break

        return total_reward

    def at_threshold(self, env, obs):
        max_capacity = env.capacity * env.num_vehicles
        total_weight = np.sum(obs.state_tuple)        
        if total_weight >= max_capacity:
            return True
        return False
        
