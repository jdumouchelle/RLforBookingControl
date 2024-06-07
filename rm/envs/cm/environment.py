import numpy as np
import matplotlib.pyplot as plt
import time
import copy

from collections import defaultdict
from typing import Dict, List, Tuple

from .observation import Observation

from .utils import *

class Environment(object):
	
	def __init__(self, 	weight_factor:float = 0.5, volume_factor:float = 0.5):
		"""
		Constructor for Environment for cargo booking and allotment requests.  

		Params
		-------------------------
		weight_factor:
			The weight capactiy ratio with respect to the total expected weight from requests.  
		volume_factor:
			The volume capactiy ratio with respect to the total expected volume from requests.  

		"""
		self.weight_factor = weight_factor
		self.volume_factor = volume_factor

		self.num_periods = 60
		self.weight_volume_ratio = 0.60
		self.cost_multiplier = 2.4
		self.expected_weight_total = 12603
		self.expected_volume_total = 7355
		self.n_items = 24
		self.n_price_ratios = 3
		self.chargeable_weights = [0.7, 1.0, 1.4]

		self.expected_weight_capacity = self.expected_weight_total * self.weight_factor
		self.expected_volume_capacity = self.expected_volume_total * self.volume_factor

		self.state = None
		
		self.weights = None
		self.volumes = None
		self.init_weights_and_volumes()
		
		self.request_probs = None
		self.init_request_probs()
		
		# init random state
		self.rng = np.random.RandomState()

		self.use_means = False
		self.reproducible = False
		self.n_final_obs = 1

		self.price_ratio_map = {0.7 : 0, 1.0 : 1, 1.4 : 2}

		return


	def set_n_final_obs(self, n_final_obs:int):
		"""
		Sets the n_final_obs of the environment.  This should only be used for 
		simulation methods and not during evaluation.  

		Params
		-------------------------
		n_final_obs:
			the number of final observations.  
		"""
		self.n_final_obs = n_final_obs
		return


	def set_use_means(self, use_means:bool):
		"""
		Sets the use_means of the environment.  This should only be used for 
		simulation methods and not during evaluation.  

		Params
		-------------------------
		use_means:
			the boolean indiciating if means should  be used when realizing the final requests.  
		"""
		self.use_means = use_means
		return


	def set_reproducible_costs(self, reproducible:bool):
		"""
		Sets the reproducible of the environment.  This should only be used when 
		evaluating and not during simulation.  

		Params
		-------------------------
		reproducible:
			the boolean indiciating if the costs realized at the end of the horizon should be 
			reproducable between instances.  
		"""
		self.reproducible = reproducible
		return
		

	def init_weights_and_volumes(self):
		"""
		Initialized the weights and volumes.
		"""
		self.weights = [50, 50, 50, 50, 100, 100, 100, 100, 200, 200, 200, 250,
						250, 300, 400, 500, 1000, 1500, 2500, 3500, 70, 70, 210, 210]
		self.volumes = [30, 29, 27, 25, 59, 58, 55, 52, 125, 119, 100, 147, 138, 
						179, 235, 277, 598, 898, 1488, 2083, 233, 17, 700, 52]
		return
	

	def init_request_probs(self):
		"""
		Initializes the request probabilties for each class of item. 

		"""
		prob_1_10 = 0.072
		prob_11_16 = 0.04
		prob_17_20 = 0.009
		prob_21_24 = 0.001

		self.request_probs = [prob_1_10]*(10-1+1) + [prob_11_16]*(16-11+1) + [prob_17_20]*(20-17+1) + [prob_21_24]*(24-21+1)
		
		return
	

	def seed(self, seed:int):
		"""
		Seeds the random number generator in the environment.

		Params
		-------------------------
		seed:
			the seed.  
		"""
		self.rng.seed(seed)
		self.seed_used = seed
		return
		

	def reset(self):
		"""
		Resets the enviornment.  

		"""
		self.state = defaultdict(int)
		self.period = 0
		
		# sample request
		request = self.sample_request(self.period)
		action_set = self.get_action_set(request)
	
		# set observation
		obs = Observation(self.state, request, action_set, self.period, self.num_periods)
		self.prev_obs = obs

		self.period += 1
		
		return obs
		
		
	def step(self, action:int):
		"""
		Takes a step in the environment.  Done by updating the state of accepted requests, 
		sampling an observation, and determining if the environment is finished running.  

		Params
		-------------------------
		action:
			the action to be taken (1 accept request, 0 reject request).
		"""

		if action not in self.prev_obs.action_set:
			print(f'Assertion Failed.  Taking invalid action. action_set={self.prev_obs.action_set}, action={action}')
			#print(action)
			#print(self.prev_obs.action_set)
			#print(self.prev_obs.request)
			#print(self.state)
			#print(self.period)

		assert(action in self.prev_obs.action_set)
		
		# update spot market accepted requests
		expected_reward = 0
		if action == 1:
			self.state[self.prev_obs.request] += 1
			expected_reward = self.get_expected_revenue(self.prev_obs.request) # FIX THIS !!!!!!!!!!
			
		# termination case
		if self.period == self.num_periods:
			if self.n_final_obs == 1:
				obs = self.sample_all_costs()
			else: # Sample a set of final observations
				obs = self.sample_n_final_obs(self.n_final_obs)

			reward = expected_reward
			done = True
			return obs, reward, done 
		
		# get next request
		request = self.sample_request(self.period)
		action_set = self.get_action_set(request)
		
		# get observation
		obs = Observation(self.state, request, action_set, self.period, self.num_periods)
		self.prev_obs = obs
		done = False
		self.period += 1    

		return obs, expected_reward, done
			
		  
	def get_action_set(self, request:Tuple):
		"""
		Gets the action set for a given request.

		Params
		-------------------------
		request:
			the request to determine the action set for.  
		"""
		if request[0] == -1:
			return (0,)
		else:
			return (0, 1)
	

	def sample_chargeable_revenue(self, period:int):
		"""
		Samples the chargeable revenue. 

		Params
		-------------------------
		period:
			the period in which the sampling is done.  
		"""
		chargeable_revs = [0.7, 1.0, 1.4, 0.0] 
		if period <= 20:
			probs = [0.7, 0.2, 0.0, 0.1]
		elif period <= 40:
			probs = [0.4, 0.2, 0.4, 0.0]
		else:
			probs = [0.0, 0.2, 0.7, 0.1]
			
		rev = self.rng.choice(chargeable_revs, p=probs)

		return rev


	def sample_request(self, period:int):
		"""
		Samples a request.  This is done by first determining the revenue ratio, 
		then sampling the item type.  

		Params
		-------------------------
		period:
			the period in which to sample the request.  

		"""
		rev = self.sample_chargeable_revenue(period)

		if rev == 0:
			return (-1, 0, period)

		item = self.rng.choice(range(24), p=self.request_probs)

		return (item, rev, period)
	
	
	def sample_costs(self, request:Tuple):
		"""
		Samples cost/weight/volume for a request at the end of the booking period.  

		Params
		-------------------------

		"""	
		# 
		if not self.use_means:
			corr = 0.80
			cv = 0.25
			sigma_w = cv * self.weights[request[0]]
			sigma_v = cv * self.volumes[request[0]]
			sigma = [[sigma_w**2, corr*sigma_w*sigma_v], [corr*sigma_w*sigma_v, sigma_v**2]]
			mu = [self.weights[request[0]], self.volumes[request[0]]]

			if not self.reproducible:
				weight, volume = self.rng.multivariate_normal(mu, sigma)
			else:
				rs = np.random.RandomState()
				rs.seed(self.seed_used + request[2])
				weight, volume = rs.multivariate_normal(mu, sigma)

		else:
			weight = self.weights[request[0]]
			volume = self.volumes[request[0]]

		chargeable_weight = max(weight, volume/self.weight_volume_ratio)
		
		price = request[1] * chargeable_weight
		cost = self.cost_multiplier * chargeable_weight
		
		d = {
			'weight' : weight,
			'volume' : volume,
			'price' : price,
			'cost' :  cost,
			'item' : request[0],
			'rate' : request[1],
			'period' : request[2],
		}
		
		return d


	def sample_all_costs(self):
		"""
		Samples end of horizon realizations for requests and capacities.  This can be made
		reproducable by setting the approraite flag.  

		"""
		assert(self.period == 60)

		final_observation = {}

		# sample capacity and volume
		if not self.use_means:
			corr = 0.80
			cv = 0.25
			sigma_w = cv * self.expected_weight_capacity
			sigma_v = cv * self.expected_volume_capacity
			sigma = [[sigma_w**2, corr*sigma_w*sigma_v], [corr*sigma_w*sigma_v, sigma_v**2]]
			mu = [self.expected_weight_capacity, self.expected_volume_capacity]

			if not self.reproducible:
				weight_cap, volume_cap = self.rng.multivariate_normal(mu, sigma)

			else:
				rs = np.random.RandomState()
				rs.seed(self.seed_used)
				weight_cap, volume_cap = rs.multivariate_normal(mu, sigma)

		else:
			weight_cap = self.expected_weight_capacity
			volume_cap = self.expected_volume_capacity

		# Fix for negative sampled weight and volume
		weight_cap = np.max([0.1, weight_cap])
		volume_cap = np.max([0.1, volume_cap])

		final_observation['weight_cap'] = weight_cap
		final_observation['volume_cap'] = volume_cap

		# sample costs and revenue for each realized item
		requests = []
		revenue = 0
		for request, count in self.state.items():
			for i in range(int(count)):
				sampled_request = self.sample_costs(request)
				requests.append(sampled_request)
				revenue += sampled_request['price']

		final_observation['requests'] = requests
		final_observation['revenue'] = revenue

		final_observation['n_requests'] = len(requests)

		# expected revenue
		expected_revenue = 0
		for request, count in self.state.items():
			expected_revenue += count * self.get_expected_revenue(request)

		final_observation['expected_revenue'] = expected_revenue
		final_observation['state'] = self.state

		# compute final observation features for linear/set models
		final_observation["linear_features"] = get_linear_features(final_observation)
		final_observation["set_features"] = get_set_features(final_observation, self.num_periods)

		return final_observation


	def sample_n_final_obs(self, n:int):
		"""
		Optimized routine for sampling n final obersvations.  This method should only be used in
		simulation.  

		Params
		-------------------------
		n:
			number of samples
		"""

		# assert that we are indeed in the simulation with the correct setting.  
		assert(self.n_final_obs > 1)
		assert(self.use_means == False)
		#assert(self.reproducible == False)

		#  capacity info
		corr = 0.80
		cv = 0.25
		sigma_w = cv * self.expected_weight_capacity
		sigma_v = cv * self.expected_volume_capacity
		sigma = [[sigma_w ** 2, corr * sigma_w * sigma_v], [corr * sigma_w * sigma_v, sigma_v ** 2]]
		mu = [self.expected_weight_capacity, self.expected_volume_capacity]

		capacity_samples = self.rng.multivariate_normal(mu, sigma, size=n)
		capacity_samples[capacity_samples < 0] = 0.1 ## Fix for negative sampled weight and volume
		weight_caps = capacity_samples[:, 0]
		volume_caps = capacity_samples[:, 1]
		
		# request info
		requests = []
		for request, count in self.state.items():
			requests += [request] * count
		n_requests = len(requests)

		# sample requests n times
		if n_requests != 0:
			corr = 0.80
			cv = 0.25
			sigma_w = cv * np.array(list(map(lambda x: self.weights[x[0]], requests)))
			sigma_v = cv * np.array(list(map(lambda x: self.volumes[x[0]], requests)))

			# compute covariance matrix
			sigma = np.zeros(2 * [2 * n_requests])
			for i in range(0, n_requests):
				sigma[i, i] = sigma_w[i] ** 2
				sigma[i+n_requests, i+n_requests] = sigma_v[i] ** 2
				sigma[i, i+n_requests] = corr * sigma_w[i] * sigma_v[i]
				sigma[i+n_requests, i] = corr * sigma_w[i] * sigma_v[i]

			mu = np.array(list(map(lambda x: self.weights[x[0]], requests)) + list(map(lambda x: self.volumes[x[0]], requests)))
			samples = self.rng.multivariate_normal(mu, sigma, size = n)
			weights = samples[:,:n_requests]
			volumes = samples[:,n_requests:]

			chargeable_weights = np.maximum(weights, volumes/self.weight_volume_ratio)

			costs = self.cost_multiplier * chargeable_weights

			ratios = np.array(list(map(lambda x: x[1], requests)))
			prices = chargeable_weights * ratios


		# get linear and set features.  Note that this is a vectorized 
		# implementation for when the number of final observations is large.
		feature_data = {
			'n_final_obs' : n,
			'n_requests' : n_requests,
			'weight_caps' : weight_caps,
			'volume_caps' : volume_caps,
		}

		if n_requests != 0:
			feature_data['weights'] = weights
			feature_data['volumes'] = volumes
			feature_data['costs'] = costs

		linear_features = get_linear_features_vectorized(feature_data)
		set_features = get_set_features_vectorized(feature_data, self.num_periods)

		revenue = 0
		expected_revenue = 0
		revenue_at_t = np.zeros(self.num_periods)
		if n_requests != 0:
			# compute average revenue    
			revenue = np.mean(np.sum(prices, axis = 1))

			# expected revenue
			expected_revenue = 0
			for request, count in self.state.items():
				expected_revenue += count * self.get_expected_revenue(request)

			# get revenue from period t onward
			mean_revenues_at_periods = np.mean(prices, axis=0)
			revenue_at_t = np.zeros(self.num_periods)
			for index, request in enumerate(self.state.keys()):
				revenue_at_t[request[2]] = mean_revenues_at_periods[index]
			
		# get revenue from t onward by reverse cumsum
		revenue_from_t_onward = np.flipud(np.flipud(revenue_at_t).cumsum())

		obs = {
			"linear_features" : linear_features,
			"set_features" : set_features,
			"revenue" : revenue,
			"expected_revenue" : expected_revenue,
			"revenue_from_t_onward" : revenue_from_t_onward,
			"feature_data" : feature_data
		} 
			
		return obs
		

	def get_expected_revenue(self, request:Tuple):
		"""
		Gets the expected revenue the given request.

		Params
		-------------------------
		request:
			the request.  

		"""
		expected_chargeable_weight = max([self.weights[request[0]], self.volumes[request[0]]/self.weight_volume_ratio])
		expected_revenue = request[1] * expected_chargeable_weight
		return expected_revenue


	def get_copy(self):
		"""
		Returns a deep copy of the environment.  This should be used
		when simulating with RL algorithms as to not modify any of
		the internal values/randomization in the environment.   

		"""
		return copy.deepcopy(self)


	def set_state(self, obs:"Observation"):
		"""
		Sets the environment.  This is done by setting the previous observation to
		be the passed obs.  This can be used when simulating with MCTS or other 
		algorithms in order set the environment to be at a specific state.  

		Params
		-------------------------
		obs:
			An observation to set the environment with.
		"""
		self.state = copy.deepcopy(obs.state)
		self.prev_obs = copy.deepcopy(obs)
		self.period = obs.period + 1