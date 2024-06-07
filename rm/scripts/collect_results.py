import argparse
import pickle

import numpy as np

import rm.params as params
from rm.utils import factory_get_path



#--------------------------------------------------------#
#                   Data collection                      #
#--------------------------------------------------------#

def get_data_generation_stats(results, cfg, args):
    """ Gets info/stats from data geneartion. """
    ml_data_fp = get_path(cfg, "ml_data", args.data_dir)

    try:
        data = pickle.load(open(ml_data_fp, 'rb'))
        results['data_generation']['time'] = data['time']
        results['data_generation']['n_samples_tr'] = len(data['tr_data'])
        results['data_generation']['n_samples_val'] = len(data['val_data'])
    
    except:
        print(f"  Failed to ML data: {ml_data_fp}")
        results['data_generation']['time'] = np.nan
        results['data_generation']['n_samples_tr'] = np.nan
        results['data_generation']['n_samples_val'] = np.nan

    return results


#--------------------------------------------------------#
#                       SL Training                      #
#--------------------------------------------------------#

def get_model_training_results(results, cfg, args, model_type):
    """ Gets training results for given model_type.  """
    
    result_fp = get_path(cfg, f"{model_type}_results", args.data_dir)
    try:
        model_results = pickle.load(open(result_fp, 'rb'))

        results['sl_results'][f'{model_type}_time'] = model_results['time']
        results['sl_results'][f'{model_type}_tr_mae'] = model_results['tr_mae']
        results['sl_results'][f'{model_type}_tr_mse'] = model_results['tr_mse']
        results['sl_results'][f'{model_type}_val_mae'] = model_results['val_mae']
        results['sl_results'][f'{model_type}_val_mse'] = model_results['val_mse']
        if model_type == "nn":
            results['sl_results'][f'{model_type}_tr_results'] = model_results['tr_results']
            results['sl_results'][f'{model_type}_val_results'] = model_results['val_results']

    except:
        print(f"  Failed to training results: {result_fp}")
        results['sl_results'][f'{model_type}_time'] = np.nan
        results['sl_results'][f'{model_type}_tr_mae'] = np.nan
        results['sl_results'][f'{model_type}_tr_mse'] = np.nan
        results['sl_results'][f'{model_type}_val_mae'] = np.nan
        results['sl_results'][f'{model_type}_val_mse'] = np.nan
        if model_type == "nn":
            results['sl_results'][f'{model_type}_tr_results'] = np.nan
            results['sl_results'][f'{model_type}_val_results'] = np.nan

    return results



#--------------------------------------------------------#
#                        Control                         #
#--------------------------------------------------------#

def get_control_results(results, cfg, args, policy_type):
    """ Gets training results for given policy_type.  """    
    control_result_fp = get_path(cfg, f"control_results/control_results_{policy_type}", args.data_dir)
    try:
        policy_results = pickle.load(open(control_result_fp, 'rb'))
        results['control_results'][f'{policy_type}_profits'] = policy_results['profits']
        results['control_results'][f'{policy_type}_profit_mean'] = policy_results['profit_mean']
        results['control_results'][f'{policy_type}_profit_std'] = policy_results['profit_std']
        results['control_results'][f'{policy_type}_total_time'] = policy_results['total_time']
        results['control_results'][f'{policy_type}_time_mean'] = policy_results['average_time']
        results['control_results'][f'{policy_type}_offline_time'] = policy_results['offline_time']

    except:
        print(f"  Failed to training results: {control_result_fp}")
        results['control_results'][f'{policy_type}_profits'] = np.nan
        results['control_results'][f'{policy_type}_profit_mean'] = np.nan
        results['control_results'][f'{policy_type}_profit_std'] = np.nan
        results['control_results'][f'{policy_type}_total_time'] = np.nan
        results['control_results'][f'{policy_type}_time_mean'] = np.nan
        results['control_results'][f'{policy_type}_offline_time'] = np.nan

    return results



#--------------------------------------------------------#
#                          Main                          #
#--------------------------------------------------------#

def main(args):
    print(f"Collecting results for {args.problem}...")

    # load config
    global get_path
    get_path = factory_get_path(args)

    cfg = getattr(params, args.problem)

    results = {}

    # add results for data generation
    results['data_generation'] = {}
    results = get_data_generation_stats(results, cfg, args)

    # add results for model training
    results['sl_results'] = {}
    results = get_model_training_results(results, cfg, args, 'lr')
    results = get_model_training_results(results, cfg, args, 'rf')
    results = get_model_training_results(results, cfg, args, 'nn')
    

    # add DQN training results
    # add control results
    results['control_results'] = {}
    if "cm" in args.problem:
        results = get_control_results(results, cfg, args, "fcfs")
        results = get_control_results(results, cfg, args, "dlp")
        results = get_control_results(results, cfg, args, "dpd")
        results = get_control_results(results, cfg, args, "dqn_linear")
        results = get_control_results(results, cfg, args, "dqn_set")

    elif "vrp" in args.problem:
        results = get_control_results(results, cfg, args, "fcfs")
        results = get_control_results(results, cfg, args, "blp")
        results = get_control_results(results, cfg, args, "blpr")
        results = get_control_results(results, cfg, args, "dqn_linear")
        results = get_control_results(results, cfg, args, "dqn_set")

    # save results
    result_fp = get_path(cfg, "combined_results", args.data_dir)
    with open(result_fp, 'wb') as p:
        pickle.dump(results, p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collects results for given problem.')

    parser.add_argument('--data_dir', type=str, default="./data/")
    parser.add_argument('--problem', type=str)
    args = parser.parse_args()

    main(args)
