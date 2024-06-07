import os
import copy
import time
import pickle
import shutil
import argparse
import numpy as np

import torch
import gym
from gym import spaces

from rm import params
from rm.utils import factory_get_path, factory_env, factory_gym_env
from rm.utils import factory_linear_model, factory_input_invariant_model



#--------------------------------------------------------#
#                          Main                          #
#--------------------------------------------------------#

def main(args):
    """ Generates data. """
    global get_path, Environment, GymEnvironment, LinearModel, IIVModel
    get_path = factory_get_path(args)
    Environment = factory_env(args)
    GymEnvironment = factory_gym_env(args)
    LinearModel = factory_linear_model(args)
    IIVModel = factory_input_invariant_model(args)

    cfg = getattr(params, args.problem)

    if 'vrp' in args.problem:
        data_dir = args.data_dir + 'vrp/random_search_dqn/'
    elif 'cm' in args.problem :
        data_dir = args.data_dir  + 'cm/random_search_dqn/'

    problem_str =  get_path(cfg, f'random_search_dqn/', args.data_dir, suffix='')
    problem_str = problem_str.split("/")[-1][1:]

    files = os.listdir(data_dir)
    files = list(filter(lambda x: f"dqn_{args.state_type}" in x, files))
    files = list(filter(lambda x: problem_str in x, files))
    files = list(filter(lambda x: f".pkl" in x, files))

    # get best result path
    best_reward = -1e5
    best_results_path = None
    for file in files:
        with open(data_dir + file, 'rb') as p:
            res = pickle.load(p)

        if res['best_reward'] > best_reward:
            best_reward = res['best_reward']
            best_results_path = file
            
    # get directories
    best_results_path = data_dir + best_results_path
    fp_results = get_path(cfg, f'dqn_{args.state_type}_results', args.data_dir)

    print("Best reward:", best_reward)
    print("Old result path:", best_results_path)
    print("New result path:", fp_results)

    shutil.copy(best_results_path, fp_results)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gets best DQN model from random search.')
    parser.add_argument('--data_dir', type=str, default="./data/", help='Data directory root.')
    parser.add_argument('--problem', type=str, help='Problem class.')
    parser.add_argument('--state_type', type=str, help='Type of state for RL model.', default="linear")

    args = parser.parse_args()

    main(args)


