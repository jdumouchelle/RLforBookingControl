import os
import numpy as np

from .product_weights import ProductWeights

from typing import Dict, List, Tuple

class VRPInstanceGenerator(object):

	
	def __init__(self, 
		num_locations:int, 
		num_products:int, 
		num_periods:int, 
		probs:Dict, 
		weights:"ProductWeights", 
		num_vehicles:int, 
		load_factor:float, 
		added_vehicle_cost:int, 
		location_bounds:Tuple[int]=(0,100)):
		"""
		Constructor for VRP instance generator.

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
		weights:
			The weights object assosiated with each item type.  
		num_vehicles:
			the number of vehicles.
		load_factor:
			The load factor which is used to determine capacity as a function of 
			the expected demand and number of vehichles.
		added_vehicle_cost:
			cost of adding each additional outsourcing vehicle.
		location_bounds:
			the bounds in which a location can be within, where the first element 
			is the lower bound and the second element is the upperbound.  
		"""
		self.num_vehicles = num_vehicles
		self.num_locations = num_locations
		self.num_products = num_products
		self.num_periods = num_periods
		self.probs = probs
		self.weights = weights
		self.load_factor = load_factor
		self.added_vehicle_cost = added_vehicle_cost
		self.location_bounds = location_bounds
		
		self.expected_demands = None
		self.capacity = None
		self.locations = None
		self.depot_location = None
		self.delivery_locations = None
		self.distance_matrix = None

		self.rng = np.random.RandomState()
		
		return


	def seed(self, seed:int):
		"""
		Seeds the instance generator for reproducibility.  

		Params
		-------------------------
		seed:
			The seed to set the random number generator with.  
		"""
		self.rng.seed(seed)
		return

	
	def generate_random(self):
		"""
		Generates a random instances based on the parameters passed.  
		"""

		# sample random locations
		locations = []
		for i in range(self.num_locations+1):
			x_coordinate = self.rng.randint(low=self.location_bounds[0], high=self.location_bounds[1])
			y_coordinate = self.rng.randint(low=self.location_bounds[0], high=self.location_bounds[1])
			locations.append(np.array((x_coordinate, y_coordinate)))
		
		self.locations = locations
		self.depot_location = self.locations[0]
		self.delivery_locations = self.locations[1:]

		self.distance_matrix = np.zeros((self.num_locations + 1, self.num_locations + 1))
		for i in range(self.num_locations + 1):
			for j in range(self.num_locations + 1):
				self.distance_matrix[i][j] = np.linalg.norm(np.array(self.locations[i]) - np.array(self.locations[j]))

		return
	
	
	def compute_capacity(self):
		"""
		Computes the capacity of the vehicles.  
		"""

		# compute mu_j
		expected_demands = np.zeros(self.num_locations+1)

		for j in range(1, self.num_locations + 1):
			for i in range(0, self.num_products):
				for t in range(0, self.num_periods):
					prob_jit = self.probs[j][i][t]
					w_i = self.weights.get_mean(i)
					expected_demands[j] += prob_jit * w_i

		self.expected_demands = expected_demands
		self.capacity = self.expected_demands.sum() / (self.load_factor * self.num_vehicles)

		# round up for numerical stability
		if self.capacity - int(self.capacity) > 1 - 1e-5:
			self.capacity = int(self.capacity + 0.1)
		self.capacity = int(self.capacity)

		return

