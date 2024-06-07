import os
import gym
import time
import pickle
import tempfile
import datetime
import argparse
import itertools

import torch
import torch.nn as nn

import numpy as np
import gurobipy as gp

import ray
from ray import tune
from ray.rllib.utils import check_env
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents import dqn
from ray.rllib.agents import ppo
from ray.tune.logger import Logger, LegacyLoggerCallback
from ray.tune import CLIReporter
from ray.tune.logger import UnifiedLogger

from rm import params
from rm.utils import factory_get_path, factory_env, factory_gym_env
from rm.utils import factory_linear_model, factory_input_invariant_model
from rm.rl_models.set_network import SetNetwork



#--------------------------------------------------------#
#                 Instance/Model loading                 #
#--------------------------------------------------------#

def get_inst_fp(args, cfg):
    """ Gets path to instance. """
    return get_path(cfg, 'inst', args.data_dir)


def load_inst(args, cfg):
    """ Loads instance.  """
    fp_env = get_inst_fp(args, cfg)
    with open(fp_env, 'rb') as p:
        env = pickle.load(p)
    return env


def get_base_model_fp(args, cfg):
    """ Gets path to base model.  """
    if "nn" == args.base_model:
        fp_model =  get_path(cfg, f'{args.base_model}', args.data_dir, suffix='.pt')
    elif args.base_model == "lr" or args.base_model == "rf":
        fp_model =  get_path(cfg, f'{args.base_model}', args.data_dir)
    else: 
        fp_model = None
    return fp_model


def load_base_model(args, cfg):
    """ Loads supervised learning model.  """
    fp_model = get_base_model_fp(args, cfg)
    if "nn" == args.base_model:
        if args.num_gpus == 0:
            base_model = torch.load(fp_model, map_location=torch.device('cpu'))
        else:
            base_model = torch.load(fp_model)

    elif args.base_model == "lr" or args.base_model == "rf":
        with open(fp_model, "rb") as p:
            base_model = pickle.load(p)
    elif args.base_model == "exact":
        base_model = None
    return base_model




#--------------------------------------------------------#
#                    Env Initialization                  #
#--------------------------------------------------------#

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


def get_env_creator(args):
    """ Returns function pointor to the environment creator.  """
    if "cm" in args.problem:
        return env_creator_cm
    elif "vrp" in args.problem:
        return env_creator_vrp


def get_env_config_vrp(args, cfg, base_env, base_model):
    """ Gets rllib env configs for VRP. """
    seed = cfg.seed 
    eval_seed = seed + 1
    train_config = {
        "base_env" : base_env,
        "ml_model" : base_model,
        "is_eval_env" : False,
        "n_eval_episodes" : 2**32-1,
        "env_seed" : seed,
        "use_oh_period" : args.use_oh_period, 
        "state_type" : args.state_type}
    eval_config = {
        "base_env" : base_env,
        "ml_model" : base_model,
        "is_eval_env" : True,
        "n_eval_episodes" : args.n_test_runs,
        "env_seed" : eval_seed,
        "use_oh_period" : args.use_oh_period, 
        "state_type" : args.state_type}
    return train_config, eval_config


def get_env_config_cm(args, cfg, base_env, base_model):
    """ Gets rllib env configs for cargo management. """
    seed = cfg.seed 
    eval_seed = seed + 1
    train_config = {
        "base_env" : base_env,
        "ml_model" : base_model,
        "cfg" : cfg,
        "reproducible_costs" : False,
        "is_eval_env" : False,
        "n_eval_episodes" : 2**32-1,
        "env_seed" : seed,
        "use_oh_period" : args.use_oh_period, 
        "state_type" : args.state_type}
    eval_config = {
        "base_env" : base_env,
        "ml_model" : base_model,
        "cfg" : cfg,
        "reproducible_costs" : True,
        "is_eval_env" : True,
        "n_eval_episodes" : args.n_test_runs,
        "env_seed" : eval_seed,
        "use_oh_period" : args.use_oh_period, 
        "state_type" : args.state_type}
    return train_config, eval_config

def get_env_config(args, cfg, base_env, base_model):
    """ Gets env configs for cargo management. """
    if "cm" in args.problem:
        return get_env_config_cm(args, cfg, base_env, base_model)
    elif "vrp" in args.problem:
        return get_env_config_vrp(args, cfg, base_env, base_model)


def print_result(result, args):
    print(f'Episode: {result["episodes_total"]}/{args.n_episodes}')
    print(f'  Timesteps: {result["timesteps_total"]}')
    print(f'  Ep stats:')
    print(f'    Mean reward: {result["episode_reward_mean"]}')
    print(f'    Max reward: {result["episode_reward_max"]}')
    print(f'    Min reward: {result["episode_reward_min"]}')
    print(f'  Eval stats:')
    print(f'    Mean reward: {result["evaluation"]["episode_reward_mean"]}')
    print(f'    Max reward: {result["evaluation"]["episode_reward_max"]}')
    print(f'    Min reward: {result["evaluation"]["episode_reward_min"]}')


def get_config_str(args):
    """ Gets unique string base on the args passed.  """
    config_str = f"bm-{args.base_model}_"
    config_str += f"lr-{args.lr}_"
    config_str += f"act-{args.activation_fn}_"
    config_str += f"duel-{args.dueling}_"
    config_str += f"dbl-{args.double_q}_"

    if args.state_type == "linear":
        if type(args.hidden_dims) is list:
            hd_str = "_".join(list(map(lambda x: str(x), args.hidden_dims)))
        else:
            hd_str = args.hidden_dims
        config_str += f"hd-{hd_str}"

    elif args.state_type == "set":
        config_str += f"e1hd-{args.iv_hidden_dim}_"
        config_str += f"e2hd-{args.iv_embed_dim1}_"
        config_str += f"e3hd-{args.iv_embed_dim2}_"
        config_str += f"chd-{args.concat_hidden_dim}"

    return config_str



#--------------------------------------------------------#
#                   Model Initialization                 #
#--------------------------------------------------------#

def get_linear_config(args, env_type, env, train_config, eval_config):
    """ Config for linear-state type. """
    activation_fn = getattr(nn, args.activation_fn)

    # compute timesteps per iteration
    timesteps_per_iteration = env.num_periods * args.test_freq

    config = {
        "env": env_type,  
        "env_config": train_config,
        
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        #"num_gpus": 1, # REMOVED FOR RAY 
        #"num_workers": 1, # REMOVED FOR RAY
        "framework": "torch",

        # number of periods
        "horizon": env.num_periods,
        
        # evaluation params
        "timesteps_per_iteration" : timesteps_per_iteration,
        "evaluation_config" : {"env_config" : eval_config},
        "evaluation_duration_unit" : "episodes",
        "evaluation_duration" : args.n_test_runs,
        "evaluation_interval" : 1,
        
        # general dqn/model params
        "lr": args.lr,
        "dueling": args.dueling,
        "double_q": args.double_q,
        "model" : {
            "fcnet_activation" : activation_fn,
        },
        
        # linear-specific parameters
        "hiddens" : args.hidden_dims,
    }

    return config


def get_set_config(args, env_type, env, train_config, eval_config):
    """ Config for linear-state type. """
    activation_fn = getattr(nn, args.activation_fn)

    # compute timesteps per iteration
    timesteps_per_iteration = env.num_periods * args.test_freq

    config = {
        "env": env_type,  
        "env_config": train_config,
        
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        #"num_gpus": 1, # REMOVED FOR RAY
        #"num_workers": 1, # REMOVED FOR RAY
        "framework": "torch",

        # number of periods
        "horizon": env.num_periods,
        
        # evaluation params
        "timesteps_per_iteration" : timesteps_per_iteration,
        "evaluation_config" : {"env_config" : eval_config},
        "evaluation_duration_unit" : "episodes",
        "evaluation_duration" : args.n_test_runs,
        "evaluation_interval" : 1,
        
        # general dqn/model params
        "dueling": args.dueling,
        "double_q": args.double_q,
        "lr": args.lr,

        # set-specific parameters
        "model" : {"custom_model" : SetNetwork,
            "fcnet_hiddens" : [args.concat_hidden_dim],
            "custom_model_config" : {
                "iv_hidden_dim" : args.iv_hidden_dim,
                "iv_embed_dim1" : args.iv_embed_dim1,
                "iv_embed_dim2" : args.iv_embed_dim2,
                "concat_hidden_dim" : args.concat_hidden_dim,
                "activation_fn" : activation_fn,
           }}          
    }

    return config


def get_dqn_config(args, env_type, env, train_config, eval_config):
    """ Gets DQN/Model config. """
    if args.state_type == "linear":
        return get_linear_config(args, env_type, env, train_config, eval_config)
    elif args.state_type == "set":
        return get_set_config(args, env_type, env, train_config, eval_config)



#--------------------------------------------------------#
#                       Logger                           #
#--------------------------------------------------------#

def custom_log_creator(custom_path, custom_str):
    """ Custom logging creator to avoid creating in root. """
    timestr = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator



#--------------------------------------------------------#
#                     Train DQN                          #
#--------------------------------------------------------#

def train(problem_config):
    """ Generates data. """
    # recover parameters from config
    args = problem_config["args"]
    cfg = problem_config["cfg"]
    data_dir = problem_config["data_dir"]
    env_type = problem_config["env_type"]
    base_env = problem_config["base_env"]
    base_model = problem_config["base_model"]
    run_name = problem_config["run_name"]
    train_config = problem_config["train_config"]
    eval_config = problem_config["eval_config"]
    fp_checkpoint = problem_config["fp_checkpoint"]
    fp_results = problem_config["fp_results"]

    # get agent config
    base_agent_config = get_dqn_config(args, env_type, base_env, train_config, eval_config)
    agent_config = dqn.DEFAULT_CONFIG.copy()
    agent_config.update(base_agent_config)

    agent = dqn.DQNTrainer(config=agent_config, env=env_type,
        logger_creator=custom_log_creator(f"{data_dir}/ray_results/", run_name))

    print(f"Training DQN-{args.state_type} model...")

    # initialize dictionary to track results
    results = {
        "tr_stats" : [],
        "best_reward" : -1e7,
        "best_ep" : -1,
        "best_checkpoint" : None,
        "params" : vars(args),
        "agent_config" : agent_config,
    }

    while True:
         # train for 1 iteration
        result = agent.train()
        
        # extract training stats from result (needed to save as pickle)
        res = {
            "episodes_total" : result["episodes_total"],
            "timesteps_total" : result["timesteps_total"],
            "episode_reward_mean" : result["episode_reward_mean"],
            "episode_reward_max" : result["episode_reward_max"],
            "episode_reward_min" : result["episode_reward_min"],
            "evaluation" : {
                "episode_reward_mean" :  result["evaluation"]["episode_reward_mean"],
                "episode_reward_max" :  result["evaluation"]["episode_reward_max"],
                "episode_reward_min" :  result["evaluation"]["episode_reward_min"],
            },
            "time" : result["time_total_s"],
        }

        results["tr_stats"].append(res)

        # if best result, save agent, then and update best-so-far
        if res["evaluation"]["episode_reward_mean"] > results["best_reward"]:

            print("New best model found.")
           
            # Save agent 
            print(f"  Saving agent ...")
            checkpoint = agent.save(fp_checkpoint)
            print(f"  Checkpoint: {checkpoint}")
        
            # Update results
            results["best_reward"] = res["evaluation"]["episode_reward_mean"] 
            results["best_ep"]  = res["episodes_total"]
            results["best_checkpoint"]  = checkpoint

            print(f"  Saving results to: {fp_results}")

            # save results
            with open(fp_results, "wb") as p:
                pickle.dump(results, p)

        tune.report(
            n_ep=res["episodes_total"],
            tr_reward=res["episode_reward_mean"],
            curr_eval_reward=res["evaluation"]["episode_reward_mean"],
            best_eval_reward=results["best_reward"] )

        # break when number of episodes exceeded
        if res["episodes_total"] > args.n_episodes:
            break

    # print final stats
    print("Total number of iters:", len(results["tr_stats"]))
    print('Max evaluation reward:', results["best_reward"] )
    print('Episode of max reward:', results["best_ep"] )

    # save results
    print(f"Saving results to {fp_results}")
    with open(fp_results, "wb") as p:
        pickle.dump(results, p)



#--------------------------------------------------------#
#                          Main                          #
#--------------------------------------------------------#

def main(args):
    """ """
    # get global variables based on problem in args
    global get_path, Environment, GymEnvironment, LinearModel, IIVModel, env_creator
    get_path = factory_get_path(args)
    Environment = factory_env(args)
    GymEnvironment = factory_gym_env(args)
    LinearModel = factory_linear_model(args)
    IIVModel = factory_input_invariant_model(args)

    # get problem config
    cfg = getattr(params, args.problem)

    # overwrite n_final_obs for cm.
    if "cm" in args.problem and args.set_n_final_obs:
        cfg.n_final_obs = args.n_final_obs

    # initialize base enviornment and sl model 
    print("Loading environment and supervised learning model ...")
    base_env = load_inst(args, cfg)
    base_model = load_base_model(args, cfg)
    print("  Done.")

    # get ray specific functions
    print("Registering as Ray environment ...")
    env_creator = get_env_creator(args)
    train_config, eval_config = get_env_config(args, cfg, base_env, base_model)

    # register rllib environment
    if "cm" in args.problem:
        env_type = "rm_cm"
    elif "vrp" in args.problem:
        env_type = "rm_vrp"
    register_env(env_type, env_creator)
    print("  Done.")

    # file paths for saving info
    print("Getting problem specific configs ...")
    working_dir = os.getcwd()
    config_str = get_config_str(args)
    if "cm" in args.problem:
        data_dir = working_dir + "/data/cm/random_search_dqn/"
    elif "vrp" in args.problem:
        data_dir = working_dir + "/data/vrp/random_search_dqn/"

    # path for dqn model saving/etc
    run_name = get_path(cfg, f'dqn_{args.state_type}', args.data_dir, suffix=f'__{config_str}__') 
    run_name = run_name.split("/")[-1]

    # path for result pickle file
    res_name = get_path(cfg, f'dqn_{args.state_type}_results', args.data_dir, suffix=f'__{config_str}__.pkl') 
    res_name = res_name.split("/")[-1]
    
    fp_checkpoint = data_dir + run_name + '/checkpoints/'
    fp_results = data_dir + res_name

    # initialize config for tune
    problem_config = {
        "args" : args,
        "cfg" : cfg,
        "data_dir" : data_dir,
        "run_name" : run_name,
        "env_type" : env_type,
        "base_env" : base_env,
        "base_model" : base_model,
        "train_config" : train_config,
        "eval_config" : eval_config,
        "fp_results" : fp_results,
        "fp_checkpoint" : fp_checkpoint,
    }
    print("  Done.")

    # Reporter info
    print("Initializing reporter ...")
    metric_columns = ["n_ep", "tr_reward", "curr_eval_reward", "best_eval_reward"]
    reporter = CLIReporter(metric_columns=metric_columns)
    print("  Done.")

    print("Initializing Ray ...")
    ray.init(local_mode=args.local_mode)
    print("  Done.")

    print("Calling tune.run ...")
    tune.run(
        train,
        resources_per_trial={
            'gpu': args.num_gpus,
            'cpu' : args.num_cpus},
        config=problem_config,
        progress_reporter=reporter,
        local_dir = data_dir,
        name=run_name,
    )
    print("  Done.")
    ray.shutdown()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains a SARSA model.')
    parser.add_argument('--data_dir', type=str, default="./data/", help='Data directory root.')
    parser.add_argument('--problem', type=str, help='Problem class.')

    # Base end-of-horizon estimator
    parser.add_argument('--base_model', type=str, help='End-of-horizon model.', default="nn")

    # Environment config
    parser.add_argument('--state_type', type=str, help='Type of state for rl.', default="set")
    parser.add_argument('--use_oh_period', type=int, help='Indicator for one-hot period encoding.', default=1)

    # General dqn/model params
    parser.add_argument('--lr', type=float, help='Learning rate.', default=0.0005)
    parser.add_argument('--activation_fn', type=str, help='Activation function.', default="ReLU")
    parser.add_argument('--dueling', type=int, help='Dueling DQN indicator.', default=1)
    parser.add_argument('--double_q', type=int, help='Double DQN indicator.', default=1)
    parser.add_argument('--train_batch_size', type=int, help='Batch size for training set.', default=32)

    # Linear-specific state parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+', help='Hidden dimensions.', default = [256, 256])

    # Set-specific state parameters
    parser.add_argument('--iv_hidden_dim', type=int, help='Dimension for req.', default=256)
    parser.add_argument('--iv_embed_dim1', type=int, help='Dimension before aggregation.', default=128)
    parser.add_argument('--iv_embed_dim2', type=int, help='Dimension after aggregation.', default=128)
    parser.add_argument('--concat_hidden_dim', type=int, help='Dimension after concatenation.', default=512)

    # Training parameters (should be fixed between runs)
    parser.add_argument('--n_episodes', type=int, help='Number of episodes.', default=15000)
    parser.add_argument('--n_test_runs', type=int, help='Number of episodes in eval.', default=100)
    parser.add_argument('--test_freq', type=int, help='Number of episodes between validation.', default=50)
    parser.add_argument('--seed', type=int, help='Seed for training.', default=1234)

    # Resources 
    parser.add_argument('--num_cpus', type=int, help='Number of cpus.', default=6)
    parser.add_argument('--num_gpus', type=int, help='Number of gpus.', default=1)

    # Local ray run (or not)
    parser.add_argument('--local_mode', type=int, help='Local mode for ray.', default=1)

    ## Set n_final_obs for CM appendix results 
    parser.add_argument('--set_n_final_obs', type=int, help='Flag to set n_final_obs', default=0)
    parser.add_argument('--n_final_obs', type=int, help='Number of final observations.', default=50)

    args = parser.parse_args()

    main(args)


