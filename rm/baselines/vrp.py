import math
import copy
import time
import itertools

import numpy as np
import gurobipy as gp

from typing import Dict, List, Tuple

from rm.envs.vrp.product_weights import ProductWeights
from rm.envs.vrp.vrp_instance_generator import VRPInstanceGenerator
from rm.envs.vrp.booking_period_env import BookingPeriodEnv
from rm.envs.vrp.vrp_solver import VRPSolver
from rm.envs.vrp.observation import Observation
from rm.envs.vrp.environment import Environment
from rm.envs.vrp.utils import *



#--------------------------------------------------------#
#               First come first serve                   #
#--------------------------------------------------------#

class FCFS(object):
    
    def __init__(self, env):
        """ Constructor for VRP. """
        self.env = env
        self.max_capacity = np.floor(self.env.capacity) * self.env.num_vehicles
        
    def get_action(self, obs):
        """ Gets next action. """
        if len(obs.action_set) == 1:
            return 0
    
        total_weight = np.sum(obs.state_tuple)        
        if total_weight >= self.max_capacity:
            action = 0
        else:
            action = 1
        
        return action



#--------------------------------------------------------#
#                    Bid-price policy                    #
#--------------------------------------------------------#

def formulate_seperation(k, alpha, beta, N, V, use_x=True):
    """ Formulates seperation problem for constraint (17) vehicle k.  """
    model_sep = gp.Model()
    model_sep.setParam("OutputFlag", 0)
    
    ### VARIABLES ###
    u, w, h = {}, {}, {}

    for i in V:
        for j in V:
            if i == j:
                continue
            u_name = f"u_{i}-{j}"
            if use_x:
                u[u_name] = model_sep.addVar(name=u_name, vtype="B", obj = alpha[i, j, k].x)
            else:
                u[u_name] = model_sep.addVar(name=u_name, vtype="B", obj = alpha[i, j, k])


    for i in V:
        w_name = f"w_{i}"
        if i == 0:
            w[w_name] = model_sep.addVar(name=w_name, vtype="B", obj = 0, lb=1, ub=1)
        else:
            w[w_name] = model_sep.addVar(name=w_name, vtype="B", obj = 0)

    for i in V:
        h_name = f"h_{i}"
        if use_x:
            h[h_name] = model_sep.addVar(name=h_name, vtype="B", obj = - beta[i, k].x)
        else:
            h[h_name] = model_sep.addVar(name=h_name, vtype="B", obj = - beta[i, k])


    ### CONSTRAINTS ###
    for i in V:
        for j in V:
            if i == j:
                continue
            model_sep.addConstr(u[f"u_{i}-{j}"] <= 2 - w[f"w_{i}"] - w[f"w_{j}"])
            model_sep.addConstr(u[f"u_{i}-{j}"] >= w[f"w_{i}"] - w[f"w_{j}"])

    # constraints for h
    eq_ = 0
    for i in V:
        eq_ += h[f"h_{i}"]
        model_sep.addConstr(h[f"h_{i}"] <= 1 - w[f"w_{i}"])
    model_sep.addConstr(eq_ == 1)
    
    # constraints for w
    eq_ = 0
    for i in V:
        eq_ += w[f"w_{i}"]
    model_sep.addConstr(eq_ <= len(V)-1)
    model_sep.addConstr(eq_ >= 1)
    
    return model_sep


def branch_and_cut(model, where):
    """ Subtour elimination constraints for integral nodes. """
    tol = 1e-5
    
    if where == gp.GRB.Callback.MIPSOL:
        incumbent_found = True
        N = model._N
        V = model._V
        K = model._K
        
        alpha = model.cbGetSolution(model._alpha)
        beta = model.cbGetSolution(model._beta)
        
        for k in K:
            
            model_sep = formulate_seperation(k, alpha, beta, N, V, use_x=False)
            model_sep.optimize()
            
            if model_sep.objVal < - tol:
                incumbent_found = False

                # get violated constraint 
                S, V_sub_S= [], []
                h = None
                for i in V:
                    if model_sep.getVarByName(f"h_{i}").x > 1 - tol:
                        h = i
                        
                    if model_sep.getVarByName(f"w_{i}").x > 1 - tol:
                        S.append(i)
                    else:
                        V_sub_S.append(i)
                        
                # add violated constraint to model
                for k in K:  # for each vehicle
                    constr_lhs = 0
                    for i in S:
                        for j in V_sub_S:
                            constr_lhs += model.getVarByName(f"alpha[{i},{j},{k}]")
                    constr_rhs = model.getVarByName(f"beta[{h},{k}]")
                    
                    model.cbLazy(constr_lhs >= constr_rhs)

        if incumbent_found:
            model._n_incumbents += 1

            if model._verbose:
                print('INCUMBENT FOUND')
                print('  N INCUMBENTS FOUND:', model._n_incumbents)
            
            if model._max_n_incumbents == -1:
                pass          

            elif model._n_incumbents >= model._max_n_incumbents:
                if model._verbose:
                    print('  FOUND MAX INCUMBENTS, TERMINATING')
                model.terminate()


class BLP(object):

    def __init__(self, reopt_periods:List[int]=[0], num_incumbent:int=2, num_incumbent_final:int=4, opt_time_interval:float=10, use_filo:bool=False, verbose=0):
        """
        Constructor for bookling limit policy (BLP) 

        Params
        -------------------------
        reopt_periods:
            Periods in which reoptimization should occur.
        num_incumbent:
            The number of incumbent solutions to terminate the PMVRP MIP instance at.
        num_incumbent_final:
            The number of incumbent solutions to terminate the VRP MIP instances at. 
        opt_time_interval:
            The time interval in which the number of incumbent solutions should be checked.
        use_filo:
            A boolean indicating if filo should be used as the VRP solver to get the routes at 
            the end of the problem.  
        """
        self.reopt_periods = reopt_periods
        self.num_incumbent = num_incumbent
        self.num_incumbent_final = num_incumbent_final
        self.opt_time_interval = opt_time_interval
        self.use_filo = use_filo
        self.verbose = verbose
        self.initial_sol = None


    def compute_initial_solution(self, env):
        """ Computes initial solution for PMVRP.  """
        t = 0
        obs = env.reset()
        self.initial_sol = self._compute_pmvrp_solution(env, obs, t)


    def run(self, env):
        """
        Runs BLP based on a environment.  

        Params
        -------------------------
        env:
            An initialized environment
        """
        t = 0
        total_reward = 0

        obs = env.reset()
        y = self.initial_sol.copy()

        while True:

            request_location = obs.location

            if request_location != 0 and y[request_location] >= 1:
                action = 1
                y[request_location] -= 1
            else:
                action = 0

            obs, reward, done = env.step(action)

            if done:
                break

            total_reward += reward
            t += 1

            # recompute y
            if t in self.reopt_periods:
                y = self._compute_pmvrp_solution(env, obs, t)

        # add opeational cost
        if self.use_filo:
            vrp_cost = self._compute_filo_vrp_solution(obs, env)
        else:
            #return self._compute_vrp_solution(env, obs, t)
            vrp_cost, _ = self._compute_vrp_solution(env, obs, t)

        profit = total_reward - vrp_cost

        return profit


    def _compute_filo_vrp_solution(self, final_obs, env):
        """
        Computes the vrp cost using filo as the VRP solver.  

        Params
        -------------------------
        final_obs:
            The observation.
        env:
            An initialize environment.  
        """

        instance_path = 'data/instances/ggm_baseline.vrp'
        costs = env.compute_vrp_cost(final_obs, instance_path)

        return costs['adjusted_operational_cost']



    def _compute_pmvrp_solution(self, env:"Environment", obs:"Observation", t:int):
        """
        Computes the solution to the PRVRP.

        Params
        -------------------------
        env:
            An intialized environment.
        obs:
            An observation which contains the current state of accepted requests.  
        t:
            The current period.  
        """
        model = self._formulate_pmvrp(env, obs, t)
        model._n_incumbents = 0
        model._max_n_incumbents = self.num_incumbent

        model.optimize(branch_and_cut)

        # recover booking limits (y)
        y = {}
        for j in range(1, env.num_locations + 1):
            y[j] = model._y[j].x

        return y


    def _compute_vrp_solution(self, env:"Environment", obs:"Observation", t:int):
        """
        Computes the solution to the VRP.

        Params
        -------------------------
        env:
            An intialized environment.
        obs:
            An observation which contains the current state of accepted requests.  
        t:
            The current period.  
        """     
        
        model = self._formulate_vrp(env, obs, t)
        model._n_incumbents = 0
        model._max_n_incumbents = self.num_incumbent_final

        model.optimize(branch_and_cut)        

        # recover objective value
        obj_val = model.objVal

        return obj_val, model


    def _set_time_limit(self, model):
        """
        Sets the time limit for the pyscipopt model.  

        Params
        -------------------------
        model:
            A pyscipopt model.
        """
        pass


    def _formulate_pmvrp(self, env:"Environment", obs:"Observation", t:int):
        """
        Formulates the PMVRP as a pyscipopt model.  

        Params
        -------------------------
        env:
            An intialized environment.
        obs:
            An observation which contains the current state of accepted requests.  
        t:
            The current period. 
        """
        model = gp.Model()
        model._verbose = self.verbose
        
        model.setParam("OutputFlag", self.verbose)
        model.setObjective(0, sense=gp.GRB.MAXIMIZE)
        model.Params.lazyConstraints = 1 # # To be used with lazy constraints

        K = list(range(env.num_vehicles))
        N = list(range(1, env.num_locations+1))
        V = list(range(env.num_locations+1))

        # DEFINE VARIABLES

        # Initialize variables
        y_ = model.addVars(N, name="y", vtype="C", obj=0, lb=0)
        alpha_ = model.addVars(V, V, K, name="alpha", vtype="B", obj=0, lb=0)
        beta_ = model.addVars(V, K, name="beta", vtype="B", obj=0, lb=0)
        q_ = model.addVars(V, K, name="q", vtype="C", obj=0, lb=0)
        
        # add variables for each location
        expected_demand_at_t = env.get_remaining_expected_demands(t)
        for j in N:
            p_j = env.prices[j][0]
            mu_j = expected_demand_at_t[j]
            y_[j].obj = p_j
            y_[j].ub = mu_j

        # add variables for each edge/vehicle in graph
        for j in V:
            for i in V:
                for k in K:
                    if j == i:
                        continue
                    loc_j = env.locations[j]
                    loc_i = env.locations[i]
                    dist_ji = np.linalg.norm(np.array(loc_j) - np.array(loc_i))
                    alpha_[j, i, k].obj = - dist_ji

        # ADD CONSTRAINTS
        # (13)
        for k in K:
            for i in V:
                V_sub_i = copy.deepcopy(V)
                V_sub_i.remove(i)
                vars_to_sum = list(map(lambda x: alpha_[i, x, k], V_sub_i))
                constraint_lhs = 0
                for var in vars_to_sum:
                    constraint_lhs += var
                constraint_rhs = beta_[i, k]
                name = f'ct_13_{i}-{k}'
                model.addConstr(constraint_lhs == constraint_rhs, name=name)

        # (14)
        for k in K:
            for i in V:
                V_sub_i = copy.deepcopy(V)
                V_sub_i.remove(i)
                vars_to_sum = list(map(lambda x: alpha_[x, i, k], V_sub_i))
                constraint_lhs = 0
                for var in vars_to_sum:
                    constraint_lhs += var
                constraint_rhs = beta_[i, k]
                name = f'ct_14_{i}-{k}'
                model.addConstr(constraint_lhs == constraint_rhs, name=name)

        # (15)
        for j in N:
            vars_to_sum = list(map(lambda x: beta_[j, x], K))
            constraint_lhs = 0
            for var in vars_to_sum:
                constraint_lhs += var
            name = f'ct_15_{j}'
            model.addConstr(constraint_lhs <= 1, name=name)

        # (16)
        vars_to_sum = list(map(lambda x: beta_[0, x], K))
        constraint_lhs = 0
        for var in vars_to_sum:
            constraint_lhs += var
        name = 'ct_16_single'
        model.addConstr(constraint_lhs <= len(K), name=name)

        # (17) minimal subtour elimination constraints
        for k in K:
            for j in N:
                name = f"ct_17_{j}-{k}"
                model.addConstr(alpha_[0,j,k] + alpha_[j,0,k] >= beta_[j,k], name=name)

        # (18)
        for j in N:
            for k in K:
                var_lhs_as_str = f"q_{j}^{k}"
                var_rhs_as_str = f"beta_{j}^{k}"
                constraint_lhs = q_[j, k]
                constraint_rhs = int(env.capacity) * beta_[j, k]
                name = f'ct_18_{k}-{k}'
                model.addConstr(constraint_lhs <= constraint_rhs, name=name)

        # (19)
        for k in K:
            vars_to_sum = list(map(lambda x: q_[x, k], N))
            constraint_lhs = 0
            for var in vars_to_sum:
                constraint_lhs += var
            constraint_rhs = int(env.capacity)
            name = f'ct_19_{k}'
            model.addConstr(constraint_lhs <= constraint_rhs, name=name)

        # (20)
        for j in N:    
            vars_to_sum = list(map(lambda x: q_[j, x], K))
            constraint_lhs = 0
            for var in vars_to_sum:
                constraint_lhs += var
            w_j = obs.state_dict[j][0]
            y_j_str = f"y_{j}"
            y_j = y_[j]
            constraint_rhs =  w_j + y_j
            name = f'ct_20_{j}'
            model.addConstr(constraint_lhs == constraint_rhs, name=name)
            
        model.update()
                        
        model._y = y_
        model._alpha = alpha_
        model._beta = beta_
        model._q = q_
        
        model._N = N
        model._K = K
        model._V = V

        return model



    def _formulate_vrp(self, env:"Environment", obs:"Observation", t:int):
        """
        Formulates the VRP as a pyscipopt model.  

        Params
        -------------------------
        env:
            An intialized environment.
        obs:
            An observation which contains the current state of accepted requests.  
        t:
            The current period. 
        """
        model = gp.Model()
        model._verbose = self.verbose

        model.setParam("OutputFlag", self.verbose)
        model.Params.lazyConstraints = 1 # # To be used with lazy constraints

        K = list(range(env.num_vehicles))
        N = list(range(1, env.num_locations+1))
        V = list(range(env.num_locations+1))

        # Initialize variables
        alpha_ = model.addVars(V, V, K, name="alpha", vtype="B", obj=0)
        beta_ = model.addVars(V, K, name="beta", vtype="B", obj=0)
        q_ = model.addVars(V, K, name="q", vtype="C", obj=0)        
        
        # add variables for each edge/vehicle in graph
        for j in V:
            for i in V:
                for k in K:
                    if j == i:
                        continue
                    loc_j = env.locations[j]
                    loc_i = env.locations[i]
                    dist_ji = np.linalg.norm(np.array(loc_j) - np.array(loc_i))
                    alpha_[j, i, k].obj = dist_ji


        # ADD CONSTRAINTS
        # (13)
        for k in K:
            for i in V:
                V_sub_i = copy.deepcopy(V)
                V_sub_i.remove(i)
                vars_to_sum = list(map(lambda j: alpha_[i, j, k], V_sub_i))
                constraint_lhs = 0
                for var in vars_to_sum:
                    constraint_lhs += var
                constraint_rhs = beta_[i, k]
                name = f'ct_13_{i}_{k}'
                model.addConstr(constraint_lhs == constraint_rhs, name=name)

        # (14)
        for k in K:
            for i in V:
                V_sub_i = copy.deepcopy(V)
                V_sub_i.remove(i)
                vars_to_sum = list(map(lambda j: alpha_[j, i, k], V_sub_i))
                constraint_lhs = 0
                for var in vars_to_sum:
                    constraint_lhs += var
                constraint_rhs = beta_[i, k]
                name = f'ct_14_{i}_{k}'
                model.addConstr(constraint_lhs == constraint_rhs, name=name)

        # (15)
        for j in N:
            vars_to_sum = list(map(lambda k: beta_[j, k], K))
            constraint_lhs = 0
            for var in vars_to_sum:
                constraint_lhs += var
            name = f'ct_15_{j}' 
            model.addConstr(constraint_lhs <= 1, name=name)

        # (16)
        vars_to_sum = list(map(lambda k: beta_[0, k], K))
        constraint_lhs = 0
        for var in vars_to_sum:
            constraint_lhs += var
        name = 'ct_16_single'
        model.addConstr(constraint_lhs <= len(K), name=name)

        # (17) minimal subtour elimination constraints
        for k in K:
            for j in N:
                name = f"ct_17_{j}-{k}"
                model.addConstr(alpha_[0,j,k] + alpha_[j,0,k] >= beta_[j,k], name=name)
                
        # (18)
        for j in N:
            for k in K:
                var_lhs_as_str = f"q_{j}^{k}"
                var_rhs_as_str = f"beta_{j}^{k}"
                constraint_lhs = q_[j, k]
                constraint_rhs = int(env.capacity) * beta_[j, k]
                name = f'ct_18_{j}_{k}'
                model.addConstr(constraint_lhs <= constraint_rhs, name=name)

        # (19)
        for k in K:
            vars_to_sum = list(map(lambda j: q_[j, k], N))
            constraint_lhs = 0
            for var in vars_to_sum:
                constraint_lhs += var
            constraint_rhs = int(env.capacity)
            name = f'ct_19_{k}'
            model.addConstr(constraint_lhs <= constraint_rhs, name=name)

        # (20)
        for j in N:
            vars_to_sum = list(map(lambda x: q_[j, x], K))
            constraint_lhs = 0
            for var in vars_to_sum:
                constraint_lhs += var
            w_j = obs.state_dict[j][0]
            constraint_rhs =  w_j 
            name = f'ct_20_{j}'
            model.addConstr(constraint_lhs == constraint_rhs, name=name)
            
        model.update()
        
        model._alpha = alpha_
        model._beta = beta_
        model._q = q_
        
        model._N = N
        model._K = K
        model._V = V

        return model
