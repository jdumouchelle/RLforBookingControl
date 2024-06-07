import os
import pickle

import gym
from gym import spaces
from collections import OrderedDict

from rm.envs.cm.utils import *
from rm.envs.cm.environment import Environment


class GymEnvironment(gym.Env):

    def __init__(self, 
                 env,
                 ml_model, 
                 n_final_obs, 
                 use_means, 
                 reproducible_costs,
                 use_oh_period=True,
                 state_type="linear",
                 exact_time_limit=300,
                 is_eval_env=False,
                 n_eval_episodes=2**32-1):

        self.env = env.get_copy()
        self.ml_model = ml_model

        self.n_final_obs = n_final_obs
        self.use_means = use_means
        self.reproducible_costs = reproducible_costs
        self.exact_time_limit = exact_time_limit
        self.use_oh_period = use_oh_period
        self.state_type = state_type

        self.is_eval_env = is_eval_env
        self.n_eval_episodes = n_eval_episodes

        self.seed_used = None

        # set up underlying environment
        self.env.set_n_final_obs(self.n_final_obs)
        self.env.set_use_means(self.use_means)
        self.env.set_reproducible_costs(self.reproducible_costs)

        # initialize gym action/observation space
        if self.state_type == "linear":
            self.initialize_linear_spaces()
        elif self.state_type == "set":
            self.initialize_set_spaces()

        # initialize input invariant feature info
        self.n_eval_resets = 0
        
    def seed(self, seed):
        """ Seeds environment. """
        self.seed_used = seed
        self.env.seed(self.seed_used)


    def seed_eval_env(self):
        """ Seeding function used for reproducible evaluation environment. """
        self.env.seed(self.seed_used + self.n_eval_resets)


    def reset(self, ):
        """ Reset the environment. """
        self.n_eval_resets += 1
        if self.is_eval_env:
            self.seed_eval_env()

        obs = self.env.reset()
        self.prev_obs = obs

        if self.state_type == "linear":
            rl_obs = self.get_linear_rl_obs(obs)
        elif self.state_type == "set":
            rl_obs = self.get_set_rl_obs(obs)

        return rl_obs


    def step(self, action):
        """ Step in the environment. """

        if len(self.prev_obs.action_set) == 1:
            action = 0

        obs, reward, done = self.env.step(action)
        self.prev_obs = obs

        # compute end-of-horizon cost
        if done:
            operational_cost = self._compute_operational_cost(obs)
            expectation_diff = self._get_expectation_diff(obs)

            reward += operational_cost + expectation_diff
            obs['operational_cost'] = operational_cost

            if self.is_eval_env:
                if self.n_eval_resets == self.n_eval_episodes:
                    self.n_eval_resets = 0

            return self.null_obs, reward, True, obs
        
        if self.state_type == "linear":
            rl_obs = self.get_linear_rl_obs(obs)
        elif self.state_type == "set":
            rl_obs = self.get_set_rl_obs(obs)

        return rl_obs, reward, False, obs.to_dict()


    def render(self):
        """ Does nothing. """
        pass


    def _compute_operational_cost(self, final_obs):
        """ Computes the operational cost at the end of the period. """

        # predict operational cost
        if self.n_final_obs == 1:
            if self.ml_model is not None:
                operational_cost = self.ml_model.predict([final_obs])[0][0]

            else:
                operational_cost = self._exact_solution(final_obs)

        else:
            assert(self.ml_model is not None)
            operational_cost = self.ml_model.predict(final_obs)
            operational_cost = np.mean(operational_cost)

        return operational_cost


    def _exact_solution(self, final_obs):
        """ Computes the exact operational cost at the end of the period. """
        if final_obs['n_requests'] == 0:
            return 0

        # get operational cost if requests exist
        model = formulate_mip_gp(final_obs)
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', self.exact_time_limit)
        model.optimize()

        operational_cost = - model.objVal

        return operational_cost


    def _compute_end_of_horizon_reward(self, final_obs):
        """ Computes the operational cost at the end of the period. """
        return self._compute_operational_cost(final_obs)


    def _get_expectation_diff(self, final_obs):
        """ Computes the difference in realized and expected revenues. """
        return final_obs["revenue"] - final_obs["expected_revenue"]


    ####
    # VECTOR STATE FEATURES
    ####
    def initialize_linear_spaces(self,):
        """ Initialize spaces for linear-type observation. """
        # action space
        self.action_space = spaces.Discrete(2)

        # features for observation space
        n_items = self.env.n_items
        n_stats = 2
        n_req_stats = 2
        n_periods = self.env.num_periods

        obs_dim = n_items + n_stats + n_req_stats
        if self.use_oh_period:
            obs_dim += n_periods
        else:
            obs_dim += 1

        # compute bounds for features
        # number of each request
        obs_ub = [20.0] * n_items
        obs_lb = [0.0] * n_items

        # number of stats (weight/volume ratios)
        obs_ub += [5.0] * n_stats
        obs_lb += [0.0] * n_stats

        # request features
        obs_ub += [n_items, 1.4]
        obs_lb += [0.0, 0.0]

        # period features
        if self.use_oh_period:
            obs_ub += [1.0] * n_periods
            obs_lb += [0.0] * n_periods
        else:
            obs_ub += [n_periods]
            obs_lb += [0.0]

        # create observation space
        obs_ub = np.float32(np.array(obs_ub))
        obs_lb = np.float32(np.array(obs_lb))

        self.observation_space = spaces.Box(low=obs_lb, high=obs_ub, dtype=np.float32)

        # create all zero observation
        self.null_obs = np.zeros(obs_dim)
                

    def get_linear_rl_obs(self, obs):
        """ Computes state for linear feature reinforcement learning models. """
        # previous request features (# previous accepted requests)
        rl_obs  = [0] * self.env.n_items 
        expected_weight = 0
        expected_volume = 0
        for request, count in obs.state.items():
            rl_obs[request[0]] += count
            expected_weight += self.env.weights[request[0]] * count
            expected_volume += self.env.volumes[request[0]] * count

        # utilized capacity and weight ratio features
        rl_obs += [expected_weight/self.env.expected_weight_capacity, expected_volume/self.env.expected_volume_capacity]

        # request features
        rl_obs += [obs.request[0] + 1, obs.request[1]]

        # period features
        if self.use_oh_period:
            period_features = [0.0] * self.env.num_periods
            period_features[obs.period] = 1.0
        else:
            period_features = [obs.period]

        rl_obs += period_features

        # state to array
        rl_obs = np.array(rl_obs)

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
        inst_dim_capacity = 4
        inst_dim_curr_req = 7
        if self.use_oh_period:
            inst_dim_period = self.env.num_periods
        else:
            inst_dim_period = 1
            
        inst_dim = inst_dim_capacity + inst_dim_period + inst_dim_curr_req
        
        # get bounds for inst
        # bounds for capacity
        inst_obs_lb = [0, 0, 0, 0]
        inst_obs_ub = [self.env.expected_weight_capacity, self.env.expected_volume_capacity, 
                       5 * self.env.expected_weight_capacity, 5 * self.env.expected_volume_capacity]
        
        # bounds for period
        if self.use_oh_period:
            inst_obs_lb += [0] * self.env.num_periods
            inst_obs_ub += [1] * self.env.num_periods
        else:
            inst_obs_lb += [0]
            inst_obs_ub += [self.env.num_periods]
        
        # bounds for request info
        request_lb = []
        request_ub = []
        
        # item class
        request_lb += [0]
        request_ub += [self.env.n_items]
        
        # price_ratio
        request_lb += [0]# [self.env.chargeable_weights[0]-0.1]
        request_ub += [self.env.chargeable_weights[-1]+0.1]
        
        # weight/volume
        request_lb += [0,0] #[np.min(self.env.weights), np.min(self.env.volumes)]
        request_ub += [np.max(self.env.weights), np.max(self.env.volumes)]
        
        # chargeable weights
        expected_chargeable_weights = np.max((np.array(self.env.weights), np.array(self.env.volumes)/self.env.weight_volume_ratio),  axis=0)
        request_lb += [0] #[np.min(expected_chargeable_weights)]
        request_ub += [np.max(expected_chargeable_weights)]
            
        # prices
        request_lb += [0]#[self.env.chargeable_weights[0] * np.min(expected_chargeable_weights)]
        request_ub += [self.env.chargeable_weights[-1] * np.max(expected_chargeable_weights)]
        
        # costs
        request_lb += [0]#[self.env.cost_multiplier * np.min(expected_chargeable_weights)]
        request_ub += [self.env.cost_multiplier * np.max(expected_chargeable_weights)]
        
        # create isnt observation space
        inst_obs_lb += request_lb
        inst_obs_lb = np.float32(np.array(inst_obs_lb))
        
        inst_obs_ub += request_ub
        inst_obs_ub = np.float32(np.array(inst_obs_ub))

        self.inst_observation_space = spaces.Box(low=inst_obs_lb, high=inst_obs_ub, dtype=np.float32)

        # observation space for requests
        req_obs_lb = np.float32(np.array(request_lb))
        req_obs_ub = np.float32(np.array(request_ub))

        req_obs_lb = np.broadcast_to(req_obs_lb, (self.env.num_periods, inst_dim_curr_req)) 
        req_obs_ub = np.broadcast_to(req_obs_ub, (self.env.num_periods, inst_dim_curr_req)) 
        
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
        x_inst = self.get_capacity_features(obs)
        x_inst += self.get_period_features(obs)
        x_inst += self.get_current_request_features(obs)
        x_inst = np.float32(np.array(x_inst))

        # Get request features
        x_req = self.get_accepted_request_features(obs)
        x_req = np.float32(np.array(x_req))
        
        # Pad reqest features
        n_reqs = x_req.shape[0]
        req_pad = self.env.num_periods - n_reqs
        x_req = np.pad(x_req, [(0, req_pad), (0, 0)] )
        

        rl_obs = OrderedDict()
        rl_obs['inst_obs'] = x_inst
        rl_obs['req_obs'] = x_req
                      
        return rl_obs


    def get_capacity_features(self, obs):
        """ Gets features for capacity"""
        expected_weight_cap = self.env.expected_weight_capacity
        expected_volume_cap = self.env.expected_volume_capacity
        
        expected_weight = 0
        expected_volume = 0
        for request, count in obs.state.items():
            expected_weight += self.env.weights[request[0]] * count
            expected_volume += self.env.volumes[request[0]] * count
            
        return [expected_weight_cap, expected_volume_cap, expected_weight, expected_volume]


    def get_period_features(self, obs):
        """ Gets the period. """
        period = obs.period
        if self.use_oh_period:
            oh_period = [0.0] * self.env.num_periods
            oh_period[period] = 1.0
            return oh_period
        else:
            return [period]
        
        
    def get_current_request_features(self, obs):
        """ Gets features for an incoming request. """        
        request = obs.request

        if request[0] == -1:
        	return [0, 0, 0, 0, 0, 0, 0]

        request_class = request[0] + 1  # ADD ONE for offset
        request_price_ratio = request[1]
        expected_weight = self.env.weights[request[0]]
        expected_volume = self.env.volumes[request[0]]
        expected_chargeable_weight = max(expected_weight, expected_volume/self.env.weight_volume_ratio)
        expected_price = request_price_ratio * expected_chargeable_weight
        expected_cost = self.env.cost_multiplier * expected_chargeable_weight
                
        request_features = [request_class, request_price_ratio, expected_weight, expected_volume,
                expected_chargeable_weight, expected_price, expected_cost]

        return request_features


    def get_accepted_request_features(self, obs):
        """ Gets features for all accepted requests. """        
        request_features = []
        for request, count in obs.state.items():
            for i in range(count):

                request_class = request[0] + 1  # ADD ONE for offset
                request_price_ratio = request[1]
                expected_weight = self.env.weights[request[0]]
                expected_volume = self.env.volumes[request[0]]
                expected_chargeable_weight = max(expected_weight, expected_volume/self.env.weight_volume_ratio)
                expected_price = request_price_ratio * expected_chargeable_weight
                expected_cost = self.env.cost_multiplier * expected_chargeable_weight
                
                feats = [request_class, request_price_ratio, expected_weight, expected_volume,
                        expected_chargeable_weight, expected_price, expected_cost]
                
                request_features.append(feats)
            
        if len(request_features) == 0:
            request_features.append([0] * 7)
            
        return request_features