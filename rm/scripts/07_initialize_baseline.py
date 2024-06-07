import sys
sys.path.append('../../envs/vrp/')

import copy
import time
import pickle
import torch
import argparse
import numpy as np

import rm.params as params
from rm.utils import factory_get_path, factory_env



#--------------------------------------------------------#
#                  Initialize Instance                   #
#--------------------------------------------------------#

def load_inst(args, cfg):
    """ Loads instance. """
    fp_env = get_path(cfg, 'inst', args.data_dir)
    with open(fp_env, 'rb') as p:
        env = pickle.load(p)
    return env




#--------------------------------------------------------#
#                  Initialize Basline                    #
#--------------------------------------------------------#

def init_cm_fcfs(args, cfg, env):
    """ Initializes FCFS baseline for cargo management.  """
    from rm.baselines.cm import FCFS
    fcfs = FCFS(env)
    fcfs.offline_time = 0
    return fcfs


def init_cm_dlp(args, cfg, env):
    """ Initializes DLP baseline for cargo management.  """
    from rm.baselines.cm import DLP
    time_ = time.time()
    dlp = DLP(env)
    dlp.init()
    time_ = time.time() - time_
    dlp.offline_time = time_
    return dlp


def init_cm_dpd(args, cfg, env):
    """ Initializes DPD baseline for cargo management.  """
    from rm.baselines.cm import DPD
    time_ = time.time()
    dpd = DPD(env)
    dpd.init()
    time_ = time.time() - time_
    dpd.offline_time = time_
    return dpd


def init_vrp_fcfs(args, cfg, env):
    """ Initializes FCFS baseline for VRP.  """
    from rm.baselines.vrp import FCFS
    fcfs = FCFS(env)
    fcfs.offline_time = 0
    return fcfs


def init_vrp_blp(args, cfg, env):
    """ Initializes BLP baseline for VRP.  """
    from rm.baselines.vrp import BLP
    if args.baseline == "blp":
        reopt_periods = [0] 
    elif args.baseline == "blpr":
        reopt_periods = [0, env.num_periods // 2]
    elif args.baseline == "blpa":
        reopt_periods = list(range(0, env.num_periods-1))#[0, env.num_periods // 2]

    blp = BLP(reopt_periods = reopt_periods, 
        verbose=args.verbose,
        num_incumbent=args.blp_n_incumbents_pmvrp,
        num_incumbent_final=args.blp_n_incumbents_pmvrp)

    time_ = time.time()
    blp.compute_initial_solution(env)
    blp.offline_time = time.time() - time_

    return blp




#--------------------------------------------------------#
#                          Main                          #
#--------------------------------------------------------#

def main(args):
    """ Generates data. """
    global get_path, Environment
    get_path = factory_get_path(args)
    Environment = factory_env(args)
    
    cfg = getattr(params, args.problem)

    # initialize enviornment
    env = load_inst(args, cfg)

    # initialize data generator
    if "cm" in args.problem:
        if args.baseline == "fcfs":
            baseline = init_cm_fcfs(args, cfg, env)
        elif args.baseline == "dlp":
            baseline = init_cm_dlp(args, cfg, env)
        elif args.baseline == "dpd":
            baseline = init_cm_dpd(args, cfg, env)
        else:
            raise Exception(f"Invalid baseline ({args.baseline}) for cargo management. ")

    # initialize data generator
    elif "vrp" in args.problem:
        if args.baseline == "fcfs":
            baseline = init_vrp_fcfs(args, cfg, env)
        elif "blp" in args.baseline:
            baseline = init_vrp_blp(args, cfg, env)
        elif "dp" in args.baseline:
            baseline = init_vrp_dp(args, cfg, env)
        else:
            raise Exception(f"Invalid baseline ({args.baseline}) for VRP. ")

    fp_baseline = get_path(cfg, f'baselines/baseline_{args.baseline}', args.data_dir)
    with open(fp_baseline, 'wb') as p:
        pickle.dump(baseline, p)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initializes baselines.')
    parser.add_argument('--data_dir', type=str, default="./data/", help='Data directory root.')
    parser.add_argument('--problem', type=str, help='Problem class.')
    
    parser.add_argument('--baseline', type=str, help='Baseline to initialize.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose flag for some baselines (BLP/BLPR).')
    parser.add_argument('--blp_n_incumbents_pmvrp', type=int, default=-1, help='Number of incumebents for PMVRP.')
    parser.add_argument('--blp_n_incumbents_vrp', type=int, default=-1, help='Number of incumebents for PMVRP.')

    args = parser.parse_args()

    main(args)

