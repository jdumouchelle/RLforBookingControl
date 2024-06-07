import numpy as np

from typing import Dict, List, Tuple

class Observation(object):
	
	def __init__(self, state, request, action_set, period, num_periods):
		self.state = state
		self.request = request
		self.action_set = action_set
		self.period = period
		self.num_periods = num_periods

	def hash(self):
		"""
		Returns the observation in a form which can be hashed.  
		"""
		# return (tuple(self.rl_state), self.request, self.action_set, self.period)
		raise Exception("Removed from current implementation.  Needs to be redone!")

	def to_dict(self):
		""" Observation as dict.  """
		obs_dict = {
			'state' : self.state,
			'request' : self.request,
			'action_set' : self.action_set,
			'period' : self.period,
			'num_periods' : self.num_periods,
		}
		return obs_dict