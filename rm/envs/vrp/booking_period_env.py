import os
import copy
import numpy as np

from .utils import get_state_as_tuple
from .product_weights import ProductWeights
from .observation import Observation

from typing import Dict, List, Tuple

class BookingPeriodEnv(object):
	
	def __init__(self, 
		num_locations:int, 
		num_products:int, 
		num_periods:int, 
		probs:Dict, 
		prices:Dict, 
		weights:"ProductWeights"):
		"""
		Constructor for BookingPeriodSimulator.  This class is used in order to simulate a booking phase
		of the problem given a set of parameters.

		Params
		-------------------------
		num_locations:
			The number of locations.  
		num_products:
			The number of products.
		num_periods:
			The number of periods in the booking period.
		probs:
			A nested dictionary containing the probability of each request.  The dictionary
			should be indexed as follows: 
				probs[j][i][t] -> probability that location j requests product type i at period t.
		prices:
			A nested dictionary containing the price assosiated with a request.  The dictionary
			should be index as follows:
				prices[j][i] -> price that location j will offer for product type i.
		weights:
			The weights object assosiated with each item type.  
		"""
		self.num_locations = num_locations
		self.num_products = num_products
		self.num_periods = num_periods
		self.probs = probs
		self.prices = prices
		self.weights = weights
		
		self.current_period = None
		self.previous_obs = None

		self.rng = np.random.RandomState()
			
		return


	def seed(self, seed:int):
		"""
		Seeds the random number generator for this class.

		Params
		-------------------------
		seed:
			The in which the random number generator is set.  
		"""
		self.rng.seed(seed)

		return
	

	def step(self, action):
		"""
		Takes an action, updates the states and returns the observation, reward, and done

		Params
		-------------------------
		action:
			The action to take in the environment.   
		"""
		
		assert(action in self.previous_obs.action_set)

		if action == 1:
			reward = self.prices[self.previous_obs.location][self.previous_obs.product]
			state_dict = copy.deepcopy(self.previous_obs.state_dict)
			state_dict[self.previous_obs.location][self.previous_obs.product] += 1
			state_tuple = get_state_as_tuple(state_dict, self.num_locations, self.num_products)

		else:
			reward = 0
			state_dict = copy.deepcopy(self.previous_obs.state_dict)
			state_tuple = get_state_as_tuple(state_dict, self.num_locations, self.num_products)

		# if the booking period is not ending
		if self.current_period < self.num_periods-1: # UPDATED
			location, product = self._sample_request()
			action_set = self._get_action_set(location)
			obs = Observation(state_dict, state_tuple, action_set, location, product, self.current_period, self.num_periods)
			done = False
		else:
			obs = Observation(state_dict, state_tuple, None, None, None, self.current_period, self.num_periods)
			done = True

		self.current_period += 1
		self.previous_obs = obs

		return obs, reward, done
	

	def reset(self):
		"""
		Reset the enviornemnt.  This returns an initial observation.  
		"""

		# initialize state dictionary
		state_dict = {}
		for j in range(1, self.num_locations + 1):
			state_dict[j] = {}
			for i in range(self.num_products):
				state_dict[j][i] = 0

		# get state as tuple
		state_tuple = get_state_as_tuple(state_dict, self.num_locations, self.num_products)

		# set period to 0
		self.current_period = 0

		# observation a request
		location, product = self._sample_request()

		# get action set
		action_set = self._get_action_set(location)

		# construct observation
		obs = Observation(state_dict, state_tuple, action_set, location, product, self.current_period, self.num_periods)

		self.previous_obs = obs
		self.current_period += 1

		return obs


	def get_copy(self):
		"""
		Returns a deep copy of the environment.  This should be used
		when simulating with RL algorithms as to not modify any of
		the internal values/randomization in the environment.   
		"""
		return copy.deepcopy(self)

	def set_state(self, obs):
		"""
		Sets the environment.  This is done by setting the previous observation to
		be the passed obs.  This can be used when simulating with MCTS or other 
		algorithms in order set the envionrment to be at a specific state.  

		Params
		-------------------------
		obs:
			An observation to set the environment with.
		"""

		self.previous_obs = obs
		self.current_period = obs.period + 1


	def _get_action_set(self, location):
		"""
		Gets action set for a specific location.

		Params
		-------------------------
		location:
			the location which is making a request (0 meaning no request).

		"""
		if location != 0:
			return (0, 1)
		return (0,)


	def _sample_request(self):
		"""
		Samples request at current period.
		"""
		# construct a matrix for the probability.
		prob_matrix = np.zeros((self.num_locations + 1, self.num_products))
		prob_matrix[0][0] = self.probs[0][self.current_period]
		for j in range(1, self.num_locations + 1):
			for i in range(0, self.num_products):
				prob_matrix[j][i] = self.probs[j][i][self.current_period]

		# build an array which can be used to index original requests from sample probability
		indexing_array = np.zeros((self.num_locations + 1) * self.num_products)
		for i in range(0, (self.num_locations + 1) * self.num_products):
			indexing_array[i] = i 

		# sample random index based on probabilites
		sample = self.rng.choice(indexing_array, p = prob_matrix.reshape(-1))

		# recover index
		j, i = np.argwhere(sample == indexing_array.reshape(self.num_locations + 1, self.num_products))[0]

		# if no requests are received, then we make no profit
		if j == 0:
			return 0, 0

		return j, i