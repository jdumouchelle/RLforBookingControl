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
        data_dir = args.data_dir + 'vrp/random_search_nn/' 
    elif 'cm' in args.problem :
        data_dir = args.data_dir  + 'cm/random_search_nn/'

    problem_str =  get_path(cfg, f'random_search_nn/', args.data_dir, suffix='')
    problem_str = problem_str.split("/")[-1][1:]

    files = os.listdir(data_dir)
    files = list(filter(lambda x: f"{args.model_type}_results" in x, files))
    files = list(filter(lambda x: problem_str in x, files))
    print(f"Number of SL models: {len(files)}")

    # get best result path
    best_score = 1e10
    best_results_path = None
    for file in files:
        with open(data_dir + file, 'rb') as p:
            res = pickle.load(p)

        if res[args.metric] < best_score:
            best_score = res[args.metric]
            best_results_path = file

    # get corresponding best model path
    best_model_path = best_results_path.split('_')
    best_model_path.remove('results')
    best_model_path = "_".join(best_model_path)
    best_model_path = best_model_path.replace('.pkl', '.pt')

    # append data directory
    best_results_path = data_dir + best_results_path
    best_model_path = data_dir + best_model_path

    fp_results = get_path(cfg, f'{args.model_type}_results', args.data_dir)
    fp_model =  get_path(cfg,  f'{args.model_type}', args.data_dir, suffix='.pt')

    print("Best score:       ", best_score)
    print("Best result path: ", best_results_path)
    print("Best model path:  ", best_model_path)

    print("Saved results to: ", fp_results)
    print("Saved model to:   ", fp_model)

    shutil.copy(best_results_path, fp_results)
    shutil.copy(best_model_path, fp_model)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recovers best random search supervised learning model.')
    parser.add_argument('--data_dir', type=str, default="./data/", help='Data directory root.')
    parser.add_argument('--problem', type=str, help='Problem class.')
    parser.add_argument('--model_type', type=str, help='Type of model.', default="nn")
    parser.add_argument('--metric', type=str, help='Metric to get model with respect to.', default="val_mae")

    args = parser.parse_args()

    main(args)


