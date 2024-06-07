import sys
sys.path.append('../../envs/vrp/')

import copy
import time
import pickle
import argparse
import numpy as np

import rm.params as params
from rm.utils import factory_get_path, factory_env, factory_data_generator


#--------------------------------------------------------#
#                   Instance loading                     #
#--------------------------------------------------------#

def load_inst(args, cfg):
    """ Loads instance. """
    fp_env = get_path(cfg, 'inst', args.data_dir)
    with open(fp_env, 'rb') as p:
        env = pickle.load(p)
    return env


#--------------------------------------------------------#
#                           Main                         #
#--------------------------------------------------------#

def main(args):
    """ Generates data. """
    global get_path, Environment, DataGenerator
    get_path = factory_get_path(args)
    Environment = factory_env(args)
    DataGenerator = factory_data_generator(args)
    
    cfg = getattr(params, args.problem)

    # initialize enviornment
    env = load_inst(args, cfg)

    # initialize data generator
    if "vrp" in args.problem:
        data_generator = DataGenerator(env, args.data_dir, n_procs=args.n_procs)
    elif "cm" in args.problem:
        data_generator = DataGenerator(env, n_procs=args.n_procs)

    data_generator.seed(cfg.seed)

    # initialize
    accept_probs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    n_train = cfg.n_train_samples // len(accept_probs)
    n_valid = cfg.n_valid_samples // len(accept_probs)

    # generate training data
    tr_time = time.time()
    tr_unique_fp = get_path(cfg, 'ml_tr_inst', args.data_dir, suffix="")
    data_generator.generate_data(accept_probs, n_train, valid=False, inst_unique_fp=tr_unique_fp)
    tr_time = time.time() - tr_time

    val_time = time.time()
    val_unique_fp = get_path(cfg, 'ml_val_inst', args.data_dir, suffix="")
    data_generator.generate_data(accept_probs, n_valid, valid=True, inst_unique_fp=val_unique_fp)
    val_time = time.time() - val_time
    
    data = {
        "tr_data" : data_generator.train_data,
        "val_data" : data_generator.valid_data,
        "tr_time" : tr_time,
        "val_time" : val_time,
        "time" : tr_time + val_time
    }

    fp_data = get_path(cfg, 'ml_data', args.data_dir)
    with open(fp_data, 'wb') as p:
        pickle.dump(data, p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data for supervised learning.')
    parser.add_argument('--data_dir', type=str, default="./data/", help='Data directory root.')
    parser.add_argument('--problem', type=str, help='Problem class.')
    parser.add_argument('--n_procs', type=int, default=16)
    parser.add_argument('--filo_exec_path', type=str, default="rm/envs/vrp/vrp_solver_filo/filo")

    args = parser.parse_args()

    main(args)

