import numpy as np
import copy

import gym
from gym import spaces
from collections import OrderedDict

from typing import Dict, List, Tuple

from .utils import *
from .environment import Environment


class GymEnvironment(gym.Env):

    def __init__(self, 
                 env,
                 ml_model,
                 use_oh_period=True,
                 state_type="linear",
                 exact_time_limit=300,
                 is_eval_env=False,
                 n_eval_episodes=2**32-1):

        self.base_env = env.copy()
        self.ml_model = ml_model
        self.exact_time_limit = exact_time_limit
        self.use_oh_period = use_oh_period
        self.state_type = state_type

        self.is_eval_env = is_eval_env
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_resets = 0

        self.seed_used = None

        self.n_locations = self.base_env.num_locations
        self.n_products = self.base_env.num_products
        self.n_periods = self.base_env.num_periods

        # initialize gym action/observation space
        if self.state_type == "linear":
            self.initialize_linear_spaces()
        elif self.state_type == "set":
            self.initialize_set_spaces()    

        
    def seed(self, seed):
        self.seed_used = seed
        self.base_env.seed(self.seed_used)

        
    def seed_eval_env(self):
        self.base_env.seed(self.seed_used + self.n_eval_resets)


    def reset(self, ):
        """ Reset the enviornment. """
        self.n_eval_resets += 1
        if self.is_eval_env:
            self.seed_eval_env()

        self.ep_reward = 0

        obs = self.base_env.reset()

        self.prev_obs = obs
        
        if self.state_type == "linear":
            rl_obs = self.get_linear_rl_obs(obs)
        elif self.state_type == "set":
            rl_obs = self.get_set_rl_obs(obs)
            
        return rl_obs


    def step(self, action):
        """ Step in the enviornment. """

        # modify to ensure valid action is taken
        if len(self.prev_obs.action_set) == 1:
            action = 0

        obs, reward, done = self.base_env.step(action)

        self.ep_reward += reward  
        self.prev_obs = obs

        # compute end-of-horizon cost
        if done:
            obs_dict = obs.to_dict()
            operational_cost = self._compute_operational_cost(obs)
            obs_dict['revenue'] = self.ep_reward
            obs_dict['operational_cost'] = operational_cost
            obs_dict['profit'] = self.ep_reward - operational_cost

            reward -= operational_cost
            self.ep_reward -= operational_cost

            rl_reward = np.float32(reward)  

            if self.is_eval_env:
                if self.n_eval_resets == self.n_eval_episodes:
                    self.n_eval_resets = 0

            return self.null_obs, rl_reward, True, obs_dict                
                  
        if self.state_type == "linear":
            rl_obs = self.get_linear_rl_obs(obs)
        elif self.state_type == "set":
            rl_obs = self.get_set_rl_obs(obs)
                   
        rl_reward = np.float32(reward)
        
        return rl_obs, rl_reward, False, obs.to_dict()


    def render(self):
        """ Does nothing. """
        pass


    def _compute_operational_cost(self, final_obs):
        """ Computes the operational cost at the end of the period. """
        # predict operational cost
        if self.ml_model is not None:
            operational_cost = self.base_env.compute_approximate_cost(final_obs, self.ml_model)
            if not np.isscalar(operational_cost):
                operational_cost = operational_cost[0]
        else:
            unique_id =  np.random.randint(2**32-1)
            unique_inst_str = f'./data/filo_eval_instances/eval_inst_{unique_id}.vrp'
            operational_cost = self.base_env.compute_exact_cost(final_obs, unique_inst_str)
        
        return operational_cost


    def _compute_end_of_horizon_reward(self, final_obs):
        """ Computes the operational cost at the end of the period. """
        return self._compute_operational_cost(final_obs)


    ####
    # VECTOR STATE FEATURES
    ####
    def initialize_linear_spaces(self,):
        """ Initializes action and observation spaces.  """
        # action space
        self.action_space = spaces.Discrete(2)

        # features for observation space
        accepted_request_dim = self.n_locations * self.n_products
        request_dim = 2

        if self.use_oh_period:
            period_dim = self.n_periods
        else:
            period_dim = 1

        obs_dim = accepted_request_dim + request_dim + period_dim

        # compute bounds for features
        # number of each request
        obs_lb = [0.0] * accepted_request_dim
        obs_ub = [self.n_periods] * accepted_request_dim
        
        # request features
        obs_lb += [0.0, 0.0]
        obs_ub += [self.n_locations, self.n_products]

        # period features
        if self.use_oh_period:
            obs_lb += [0.0] * self.n_periods
            obs_ub += [1.0] * self.n_periods
        else:
            obs_lb += [0.0]
            obs_ub += [self.n_periods]
            
        # create observation space
        obs_lb = np.float32(np.array(obs_lb))
        obs_ub = np.float32(np.array(obs_ub))
       
        self.observation_space = spaces.Box(low=obs_lb, high=obs_ub, dtype=np.float32)

        # create all zero observation
        self.null_obs = np.zeros(obs_dim)

        
    def get_linear_rl_obs(self, obs):
        """ Computes state for linear feature reinforcement learning models. """
        # features for accepted requests
        rl_obs = list(obs.state_tuple)

        # features for current request
        rl_obs += [obs.location, obs.product]

        # features for period
        if self.use_oh_period:
            period_feats = [0.0] * self.n_periods
            period_feats[obs.period] = 1.0
        else:
            period_feats = [obs.period]

        rl_obs += period_feats

        # state to array
        rl_obs = np.float32(np.array(rl_obs))

        return rl_obs
    
    
    ####
    # INPUT INVARIANT STATE FEATURES!
    ####
    def initialize_set_spaces(self):
        """ Initialize spaces for set-type observation. """
        # action space
        self.action_space = spaces.Discrete(2)
        # observation space for instance information
        # expected_weight_cap, expected_volume_cap, expected_weight, expected_volume
        # period
        # request_class, request_price_ratio, expected_weight, expected_volume, expected_chargeable_weight, expected_price, expected_cost
        inst_dim_prob = 3 # total capacity, utilized capacity, num vehicles
        inst_dim_req = 4 # loc index, product type, x-loc, y-loc
        if self.use_oh_period:
            inst_dim_period = self.n_periods
        else:
            inst_dim_period = 1
            
        inst_dim = inst_dim_prob + inst_dim_period + inst_dim_req
        
        # get bounds for inst
        # bounds for capacity
        inst_obs_lb = [0, 0, 0]
        inst_obs_ub = [self.base_env.num_vehicles * self.base_env.capacity, 4 * self.base_env.capacity, self.base_env.num_vehicles]
        
        # bounds for period
        if self.use_oh_period:
            inst_obs_lb += [0] * self.n_periods
            inst_obs_ub += [1] * self.n_periods
        else:
            inst_obs_lb += [0]
            inst_obs_ub += [self.n_periods]
            
        # bounds for request info
        request_lb = []
        request_ub = []
        
        # loc index
        request_lb += [0]
        request_ub += [self.n_locations + 1]
        
        # product index
        request_lb += [0]
        request_ub += [self.n_products + 1]
        
        # loc coordinates        
        request_lb += [0, 0]
        request_ub += [np.max(np.array(self.base_env.locations)[:,0]), np.max(np.array(self.base_env.locations)[:,0])]
            
        # create inst observation space
        inst_obs_lb += request_lb
        inst_obs_lb = np.float32(np.array(inst_obs_lb))
        
        inst_obs_ub += request_ub
        inst_obs_ub = np.float32(np.array(inst_obs_ub))

        self.inst_observation_space = spaces.Box(low=inst_obs_lb, high=inst_obs_ub, dtype=np.float32)

        # observation space for requests
        req_obs_lb = np.float32(np.array(request_lb))
        req_obs_ub = np.float32(np.array(request_ub))

        req_obs_lb = np.broadcast_to(req_obs_lb, (self.n_periods, inst_dim_req)) 
        req_obs_ub = np.broadcast_to(req_obs_ub, (self.n_periods, inst_dim_req)) 
        
        self.req_observation_space = spaces.Box(low=req_obs_lb, high=req_obs_ub, dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'inst_obs': self.inst_observation_space,
            'req_obs': self.req_observation_space,
        })

        # create all zero observation
        self.null_obs = OrderedDict({
            'inst_obs' : np.zeros(self.inst_observation_space.shape),
            'req_obs' :  np.zeros(self.req_observation_space.shape),
        })
        
        
    def get_set_rl_obs(self, obs):
        """ Gets input for input invariant networks.  """
        # Get instance features
        x_inst = self.get_problem_features(obs)
        x_inst += self.get_period_features(obs)
        x_inst += self.get_current_request_features(obs)
        x_inst = np.float32(np.array(x_inst))

        # Get request features
        x_req = self.get_accepted_request_features(obs)
        x_req = np.float32(np.array(x_req))
        
        # Pad request features
        n_reqs = x_req.shape[0]
        req_pad = self.n_periods - n_reqs
        x_req = np.pad(x_req, [(0, req_pad), (0, 0)] )
        

        rl_obs = OrderedDict()
        rl_obs['inst_obs'] = x_inst
        rl_obs['req_obs'] = x_req
                      
        return rl_obs


    def get_problem_features(self, obs):
        """ Gets features for problem. """
        total_cap = self.base_env.capacity * self.base_env.num_vehicles
        util_cap = np.sum(obs.state_tuple)        
        return [total_cap, util_cap, self.base_env.num_vehicles]

        
    def get_period_features(self, obs):
        """ Gets the period. """
        period = obs.period
        if self.use_oh_period:
            oh_period = [0.0] * self.base_env.num_periods
            oh_period[period] = 1.0
            return oh_period
        else:
            return [period]
        
        
    def get_current_request_features(self, obs):
        """ Gets features for an incoming request. """   
        loc = obs.location
        prod = obs.product
        
        if loc == -1:
            return [0, 0, 0, 0]

        request_loc = loc + 1  # ADD ONE for offset
        request_prod = prod + 1
        
        request_x = self.base_env.locations[loc][0]
        request_y = self.base_env.locations[loc][1]

        request_features = [request_loc, request_prod, request_x, request_y]

        return request_features


    def get_accepted_request_features(self, obs):
        """ Gets features for all accepted requests. """        
        request_features = []
        for loc, count in enumerate(obs.state_tuple):
            for i in range(count):
                request_loc = loc + 1  # ADD ONE for offset
                request_prod = 1 
                request_x = self.base_env.locations[loc][0]
                request_y = self.base_env.locations[loc][1]

                feats = [request_loc, request_prod, request_x, request_y]
                
                request_features.append(feats)
            
        if len(request_features) == 0:
            request_features.append([0] * 4)
            
        return request_features