import math
import pickle
import argparse
import numpy as np

import rm.params as params
from rm.utils import factory_get_path, factory_env
from rm.envs.vrp.utils import get_weights, get_prices, get_request_probs


#--------------------------------------------------------#
#               VRP Problem Initialization               #
#--------------------------------------------------------#

def initialize_vrp_problem(cfg, args):
    """ Initilizes problem based on config. """

    # booking period parameters
    n_locations = cfg.n_locations
    n_products = cfg.n_products
    n_periods = cfg.n_periods

    # vrp instance parameters
    n_vehicles = cfg.n_vehicles
    load_factor = cfg.load_factor
    added_vehicle_cost = cfg.added_vehicle_cost
    location_bounds = cfg.location_bounds

    # get prices/weights/probs
    prices = get_prices(cfg)
    weights = get_weights(cfg)
    probs = get_request_probs(cfg)

    # output path for intermediate filo files
    outpath = args.data_dir + 'vrp/filo_data/output/'

    env = Environment()

    env.set_booking_period_parameters(
        num_locations = cfg.n_locations, 
        num_products = cfg.n_products, 
        num_periods = cfg.n_periods, 
        probs = probs, 
        prices = prices, 
        weights = weights)

    env.set_vrp_parameters(
        num_vehicles = cfg.n_vehicles, 
        load_factor = cfg.load_factor, 
        added_vehicle_cost = cfg.added_vehicle_cost,
        location_bounds = cfg.location_bounds)

    env.set_solver_parameters(
        exec_path = args.filo_exec_path,
        outpath = outpath,
        parser = cfg.parser,                    
        tolerance = cfg.tolerance,              
        granular_neighbors = cfg.granular_neighbors,
        cache = cfg.cache,
        routemin_iterations = cfg.routemin_iterations,    
        coreopt_iterations = cfg.coreopt_iterations,   
        granular_gamma_base = cfg.granular_gamma_base,    
        granular_delta = cfg.granular_delta,     
        shaking_lower_bound = cfg.shaking_lower_bound,   
        shaking_upper_bound = cfg.shaking_upper_bound,
        seed = cfg.seed)

    env.initialize(cfg.seed)

    return env


#--------------------------------------------------------#
#        Cargo Management Problem Initialization         #
#--------------------------------------------------------#

def initialize_cm_problem(cfg, args):
    """ Initilizes cargo management problem based on config. """
    env = Environment(
        weight_factor = cfg.weight_factor,
        volume_factor = cfg.volume_factor)
    return env


#--------------------------------------------------------#
#                           Main                         #
#--------------------------------------------------------#

def main(args):
    """
    Generates env for specified scenario and store in pickle file.

    Params
    -------------------------
    args:
        arguements which are parsed through parse_args()
    """
    global get_path, Environment
    get_path = factory_get_path(args)
    Environment = factory_env(args)

    cfg = getattr(params, args.problem)

    # generate problem instance
    if "vrp" in args.problem:
        env = initialize_vrp_problem(cfg, args)

    elif "cm" in args.problem:
        env = initialize_cm_problem(cfg, args)

    else: 
        raise Exception(f"Problem {args.problem} is not valid.")

    # save env 
    fp_env = get_path(cfg, 'inst', args.data_dir)
    with open(fp_env, 'wb') as p:
        pickle.dump(env, p)

    return  


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate instances for VRP problem.')
    parser.add_argument('--data_dir', type=str, default="./data/", help='Data directory root.')
    parser.add_argument('--filo_exec_path', type=str, default="rm/envs/vrp/vrp_solver_filo/filo")
    parser.add_argument('--problem', type=str, help='Problem class.')

    args = parser.parse_args()
    
    main(args)
