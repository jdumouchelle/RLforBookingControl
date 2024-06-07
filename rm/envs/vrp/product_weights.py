import os
import numpy as np

from typing import Dict, List, Tuple

class ProductWeights(object):
	
	def __init__(self, num_products:int, sampling_protocol:str='constant', seed:int = 1234):
		"""
		Constructor for the ProductWeights class. 

		Params
		-------------------------
		num_products:
			The number of products.
		sampling_protocol:
			The sampling protocol to be used when sampling weights.  
			Currently one of 'constant' or 'normal'
		seed:
			The seed to set to ensure reproducability.  
		"""
		self.num_products = num_products
		self.sampling_protocol = sampling_protocol
		self.weight_means = None

		self.seed = seed
		self.rng = np.random.RandomState(self.seed)

	def set_means(self, weight_means:np.ndarray):
		"""
		Sets the means for the product types.  

		Params
		-------------------------
		weight_means:
			An array contianing the means of each product type.  
		"""
		self.weight_means = weight_means
	
	def get_mean(self, i:int):
		"""
		Gets the mean of the specified product.

		Params
		-------------------------
		i: 
			the product index.
		"""
		return self.weight_means[i] 
	
	def sample(self, i:int):
		"""
		Samples via the protocol specified in the constructor.

		Params
		-------------------------
		i: 
			the product index.
		"""
		if self.sampling_protocol == 'constant':
			return self.sample_constant(i)
		
		elif self.sampling_protocol == 'normal':
			return self.sample_normal(i, sigma=1.0)
		
		else: 
			raise('Not a valid sampling method') 
	
	def sample_normal(self, i:int, sigma:float=1.0):
		"""
		Samples weights from a normal distribution.

		Params
		-------------------------
		i: 
			the product index.
		sigma:
			the standard deviation to be used in sampling.  
		"""
		w_i = self.rng.normal(self.weight_means[i], sigma)
		return w_i
	
	def sample_constant(self, i:int):
		"""
		Samples weights from a constant value.

		Params
		-------------------------
		i: 
			the product index. 
		"""
		w_i = self.weight_means[i]
		return w_i
	
	