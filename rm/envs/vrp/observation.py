import numpy as np

from typing import Dict, List, Tuple

class Observation(object):

	def __init__(self, 
		state_dict:Dict, 
		state_tuple:Tuple, 
		action_set:Tuple, 
		location:int, 
		product:int, 
		period:int,
		num_periods:int):
		"""
		Constructor for Observation.  

		Params
		-------------------------
		state_dict: 
			The state as a dictionary.
		state_typle: 
			The state as a tuple.
		action_set: 
			The set of possible actions at the state.
		location: 
			The location which is requesting.
		product: 
			The product type which is being requested.
		period: 
			The period.
		"""
		self.state_dict = state_dict
		self.state_tuple = state_tuple
		self.action_set = action_set
		self.location = location
		self.product = product
		self.period = period
		self.num_periods = num_periods


	def hash(self):
		"""
		Returns the observation in a form which can be hashed.  
		"""
		#return (self.state_tuple, self.action_set, self.location, self.product, self.period)
		raise Exception("Removed from current implementation.  Needs to be redone!")


	def to_dict(self):
		""" 
		Returns final state information
		"""
		d = {
			'state_dict' : self.state_dict,
			'state_tuple' : self.state_tuple,
			'action_set' : self.action_set,
			'location' : self.location,
			'product' : self.product,
			'period' : self.period,
			'num_periods' : self.num_periods,
		}

		return d