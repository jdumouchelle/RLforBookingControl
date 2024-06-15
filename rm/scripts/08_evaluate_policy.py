import sys
sys.path.append('../../envs/vrp/')

import os
import copy
import time
import pickle
import torch
import argparse
import numpy as np

import rm.params as params
from rm.utils import factory_get_path, factory_env, factory_policy_evaluator, factory_gym_env
from rm.rl_models.set_network import SetNetwork

import ray
from ray.rllib.agents import dqn
from ray.tune.registry import register_env



#--------------------------------------------------------#
#                   Initialize Envs                      #
#--------------------------------------------------------#

def get_env_creator(args):
    """ RLLib specific function.  Returns function pointor to the environment creator.  """
    def env_creator_vrp(env_config):
        """ Env creator function for VRP. """
        genv = GymEnvironment(
            env = env_config["base_env"],
            ml_model = env_config["ml_model"],
            is_eval_env = env_config["is_eval_env"],
            n_eval_episodes = env_config["n_eval_episodes"],
            use_oh_period = env_config["use_oh_period"], 
            state_type = env_config["state_type"],)
        genv.seed(env_config["env_seed"])
        return genv

    def env_creator_cm(env_config):
        """ Env creator function for cargo management. """
        genv = GymEnvironment(
            env = env_config["base_env"],
            ml_model = env_config["ml_model"], 
            n_final_obs = env_config["cfg"].n_final_obs, 
            use_means = env_config["cfg"].deterministic, 
            reproducible_costs = env_config["reproducible_costs"], 
            use_oh_period = env_config["use_oh_period"],
            state_type = env_config["state_type"],
            exact_time_limit = 3600,
            is_eval_env = env_config["is_eval_env"],
            n_eval_episodes = env_config["n_eval_episodes"])
        genv.seed(env_config["env_seed"])
        return genv

    if "cm" in args.problem:
        return env_creator_cm
    elif "vrp" in args.problem:
        return env_creator_vrp




#--------------------------------------------------------#
#              Initialize Instance/Imports               #
#--------------------------------------------------------#

def load_inst(args, cfg):
    """ Loads instance. """
    fp_env = get_path(cfg, 'inst', args.data_dir)
    with open(fp_env, 'rb') as p:
        env = pickle.load(p)
    return env


def load_imports(args, cfg):
    """ Loads imports for problem class learning model.  """
    if "cm" in args.problem:
        global FCFS, DLP, DPD
        from rm.baselines.cm import FCFS, DLP, DPD

    elif "vrp" in args.problem:
        global BLP
        from rm.baselines.vrp import BLP




#--------------------------------------------------------#
#                   Initialize Policies                  #
#--------------------------------------------------------#

def init_rllib_policy(args, cfg):
    """ Initialize an rllib policy (DQN only). """
    fp = get_path(cfg, f'{args.policy}_results', args.data_dir, suffix=f'.pkl')
    res = pickle.load(open(fp, "rb"))

    # initialize dqn agent
    agent = dqn.DQNTrainer(config=res["agent_config"], env=env_type)
    agent.use_oh_period = res["params"]["use_oh_period"] # ADD INFO FOR OH
 
    best_checkpoint = res["best_checkpoint"]

    # exception handling incase running DQN model on different device
    if not os.path.exists(best_checkpoint):
        print(f"Modifying checkpoint dir to {args.data_dir}.")
        print(f"  Note that this may not work for relative paths, absolute paths are preferred.")

        best_checkpoint = best_checkpoint.split("data/")[-1]
        best_checkpoint = args.data_dir + best_checkpoint

    agent.restore(best_checkpoint)
    agent.offline_time = res["tr_stats"][-1]["time"]

    return agent


def load_policy(args, cfg):
    """ Loads policy.  """
    if args.policy in baseline_set:
        fp_policy = get_path(cfg, f'baselines/baseline_{args.policy}', args.data_dir)
        with open(fp_policy, 'rb') as p:
            policy = pickle.load(p)

    elif args.policy in rllib_set:
        policy = init_rllib_policy(args, cfg)

    return policy


def evaluate_cm_policy(args, cfg, env, policy):
    """ Evaluates a policy for cargo management.  """
    pe = PolicyEvaluator(env, args.n_iters, args.seed, print_stats=True)

    if args.policy in rllib_set:
        state_type = args.policy.split("_")[-1]
        results = pe.evaluate_control(policy, use_gym=True, is_dqn=True, state_type=state_type, use_threshold=args.use_threshold)
        results["offline_time"] = policy.offline_time

    else:
        policy.env = env # fix for resetting environment
        results = pe.evaluate_control(policy, use_gym=False, is_dqn=False)
        results["offline_time"] = policy.offline_time

    return results


def evaluate_vrp_policy(args, cfg, env, policy):
    """ Evaluates policy for VRP.  """
    pe = PolicyEvaluator(args.data_dir, env, args.n_iters, args.seed, print_stats=True)

    results = {}

    print("EVALUATING:", args.policy)

    if args.policy in rllib_set:
        state_type = args.policy.split("_")[-1]
        results = pe.evaluate_control(policy, use_gym=True, is_dqn=True, state_type=state_type, use_threshold=args.use_threshold)
        results["offline_time"] = policy.offline_time

    elif "fcfs" in args.policy:
        results = pe.evaluate_control(policy, use_gym=False, is_dqn=False, state_type="linear")
        results["offline_time"] = np.nan

    elif "blp" in args.policy:
        results = pe.evaluate_baseline(policy)
        results["offline_time"] = policy.offline_time

    return results




#--------------------------------------------------------#
#                          Main                          #
#--------------------------------------------------------#

def main(args):
    """ Generates data. """
    global get_path, Environment, PolicyEvaluator, GymEnvironment
    get_path = factory_get_path(args)
    Environment = factory_env(args)
    GymEnvironment = factory_gym_env(args)
    PolicyEvaluator = factory_policy_evaluator(args)
    
    # global baseline_set, rl_set, rllib_set
    global baseline_set, rllib_set
    baseline_set = ["dp", "dp_nn", "fcfs", "blp", "blpr", "blpa", "dlp", "dpd"]
    rllib_set = ["dqn_linear", "dqn_set"]

    use_ray = args.policy in rllib_set

    if use_ray:
        # Initialize ray/rllib is agent is in policy
        global env_type
        ray.init()
        env_creator = get_env_creator(args)
        if "cm" in args.problem:
            env_type = "rm_cm"
        elif "vrp" in args.problem:
            env_type = "rm_vrp"
        register_env(env_type, env_creator)

    cfg = getattr(params, args.problem)

    # overwrite n_final_obs for cm appendix results.
    if "cm" in args.problem and args.set_n_final_obs:
        cfg.n_final_obs = args.n_final_obs

    load_imports(args, cfg)

    # initialize enviornment
    env = load_inst(args, cfg)

    # load all control policy
    policy = load_policy(args, cfg)

    # evaluate control policy
    if "cm" in args.problem:
        results = evaluate_cm_policy(args, cfg, env, policy)

    elif "vrp" in args.problem:
        results = evaluate_vrp_policy(args, cfg, env, policy)  

    # Print summary of results
    print("\nMean Profits:")
    print(f"  {args.policy}:", results['profit_mean'])

    print("\nMean Online Time:")
    print(f"  {args.policy}:", results['average_time'])

    print("\nOffline Times:")
    print(f"  {args.policy}:", results['offline_time'])

    # Shutdown ray if needed.
    if use_ray:
        ray.shutdown()

    fp_results = get_path(cfg, f'control_results/control_results_{args.policy}', args.data_dir)
    with open(fp_results, 'wb') as p:
        pickle.dump(results, p)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initializes baselines.')
    parser.add_argument('--data_dir', type=str, default="./data/", help='Data directory root.')
    parser.add_argument('--problem', type=str, help='Problem class.')
    
    parser.add_argument('--policy', type=str, help='Policies to evaluate')
    parser.add_argument('--use_threshold', type=int, default=1, help='Indcator to use threshold for policy evaluation for DQN.')
    parser.add_argument('--n_iters', type=int, default=1000, help='Number of episodes to evaluate over.')
    parser.add_argument('--seed', type=int, default=0, help='Seed.')

    # args for n_final_obs experiements in appendix
    parser.add_argument('--n_final_obs', type=int, help='Number of final observations (for DQN with CM only).', default=1)
    parser.add_argument('--set_n_final_obs', type=int, help='Flag to set number of final observations (for DQN with CM only).', default=0)

    args = parser.parse_args()

    main(args)

