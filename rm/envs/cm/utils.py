import torch
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
from collections import defaultdict

from typing import Dict, List, Tuple


def get_set_data(dataset, max_requests=60):
	""" Gets set representation for features.  """
	
	# instance information
	x_inst = list(map(lambda x: x['set_features']['x_inst'], dataset))
	x_inst = torch.stack(x_inst)

	# requuest information
	x_req = list(map(lambda x: x['set_features']['x_req'], dataset))
	x_req = torch.stack(x_req)

	# request information
	x_n_req = list(map(lambda x: x['set_features']['x_n_req'], dataset))
	x_n_req = torch.stack(x_n_req)

	# get targets, if in dataset
	y = None
	if 'operational_cost' in dataset[0]:
		y = np.array(list(map(lambda x: x['operational_cost'], dataset))).reshape(-1,1)
		y = torch.from_numpy(y).float()

	return x_inst, x_req, x_n_req, y


def get_linear_data(dataset):
	""" Gets data for scikit-learn models.  """
	x = list(map(lambda x: np.array(list(x['linear_features'].values())), dataset))
	
	y = None
	if 'operational_cost' in dataset[0]:
		y = list(map(lambda x: x['operational_cost'], dataset))

	return x, y


def get_linear_features(obs:Dict):
	"""
	Gets features for the final observation.  

	Params
	-------------------------
	obs:
		The observation at the end of the booking period. 
	"""
	requests = obs["requests"]
	
	features = {}
	
	features["n_items"] = len(requests)
	features["cap_w"] = obs["weight_cap"]
	features["cap_v"] = obs["volume_cap"]

	if len(requests) == 0:
		weights = [0]
	else:
		weights = list(map(lambda x: x["weight"], requests))

	features["min_w"] = np.min(weights)
	features["max_w"] = np.max(weights)
	features["mean_w"] = np.mean(weights)
	features["median_w"] = np.median(weights)
	features["std_w"] = np.std(weights)
	features["q25_w"] = np.quantile(weights, 0.25)
	features["q75_w"] = np.quantile(weights, 0.75)
	features["sum_w"] = np.sum(weights)

	if len(requests) == 0:
		volumes = [0]
	else:
		volumes = list(map(lambda x: x["volume"], requests))

	features["min_v"] = np.min(volumes)
	features["max_v"] = np.max(volumes)
	features["mean_v"] = np.mean(volumes)
	features["median_v"] = np.median(volumes)
	features["std_v"] = np.std(volumes)
	features["q25_v"] = np.quantile(volumes, 0.25)
	features["q75_v"] = np.quantile(volumes, 0.75)
	features["sum_v"] = np.sum(volumes)

	if len(requests) == 0:
		costs = [0]
	else:
		costs = list(map(lambda x: x["cost"], requests))

	features["min_c"] = np.min(costs)
	features["max_c"] = np.max(costs)
	features["mean_c"] = np.mean(costs)
	features["median_c"] = np.median(costs)
	features["std_c"] = np.std(costs)
	features["q25_c"] = np.quantile(costs, 0.25)
	features["q75_c"] = np.quantile(costs, 0.75)
	features["sum_c"] = np.sum(costs)

	return features


def get_set_features(obs:Dict, max_requests:int):
	"""
	Gets features for the final observation.  

	Params
	-------------------------
	obs:
		The observation at the end of the booking period. 
	max_requests:
		The number of periods.
	"""    
	def get_req_featues(obs):
		if obs['n_requests'] == 0:
			return [[0] * 3]
		req_features = []
		for request in obs['requests']:
			req_features.append([request['weight'], request['volume'], request['cost']]) 
		return req_features
	
	def pad_tensor(t, max_requests):
		pd = (0, 0, 0, max_requests - t.shape[0])
		p = torch.nn.ZeroPad2d(pd)
		return p(t)

	x_inst = [obs['weight_cap'], obs['volume_cap'], obs['n_requests']]
	x_inst = torch.FloatTensor(x_inst)

	x_req = get_req_featues(obs)
	x_req = torch.from_numpy(np.array(x_req)).float()
	x_req = pad_tensor(x_req, max_requests)

	x_n_req = obs['n_requests']
	x_n_req = torch.ones((1)) * x_n_req

	set_features = {
		'x_inst' : x_inst,
		'x_req' : x_req,
		'x_n_req' : x_n_req 
	}

	return set_features


def get_linear_features_vectorized(feature_data:Dict):
	""" 
	Gets features for the final observation.  Vectorized implementation for 
	when the number of final observations > 1.
	"""
	features = np.zeros((feature_data['n_final_obs'], 8 * 3 + 3))

	features[:, 0] = feature_data['n_requests']
	features[:, 1] = feature_data['weight_caps']
	features[:, 2] = feature_data['volume_caps']

	if feature_data['n_requests'] != 0:
		idx = 3
		features[:, idx+0] = np.min(feature_data['weights'], axis=1)
		features[:, idx+1] = np.max(feature_data['weights'], axis=1)
		features[:, idx+2] = np.mean(feature_data['weights'], axis=1)
		features[:, idx+3] = np.median(feature_data['weights'], axis=1)
		features[:, idx+4] = np.std(feature_data['weights'], axis=1)
		features[:, idx+5] = np.quantile(feature_data['weights'], 0.25, axis=1)
		features[:, idx+6] = np.quantile(feature_data['weights'], 0.75, axis=1)
		features[:, idx+7]= np.sum(feature_data['weights'], axis=1)

	if feature_data['n_requests'] != 0:
		idx = 3 + 8
		features[:, idx+0] = np.min(feature_data['volumes'], axis=1)
		features[:, idx+1] = np.max(feature_data['volumes'], axis=1)
		features[:, idx+2] = np.mean(feature_data['volumes'], axis=1)
		features[:, idx+3] = np.median(feature_data['volumes'], axis=1)
		features[:, idx+4] = np.std(feature_data['volumes'], axis=1)
		features[:, idx+5] = np.quantile(feature_data['volumes'], 0.25, axis=1)
		features[:, idx+6] = np.quantile(feature_data['volumes'], 0.75, axis=1)
		features[:, idx+7]= np.sum(feature_data['volumes'], axis=1)
		
	if feature_data['n_requests'] != 0:
		idx = 3 + 8 + 8
		features[:, idx+0] = np.min(feature_data['costs'], axis=1)
		features[:, idx+1] = np.max(feature_data['costs'], axis=1)
		features[:, idx+2] = np.mean(feature_data['costs'], axis=1)
		features[:, idx+3] = np.median(feature_data['costs'], axis=1)
		features[:, idx+4] = np.std(feature_data['costs'], axis=1)
		features[:, idx+5] = np.quantile(feature_data['costs'], 0.25, axis=1)
		features[:, idx+6] = np.quantile(feature_data['costs'], 0.75, axis=1)
		features[:, idx+7]= np.sum(feature_data['costs'], axis=1)

	return features


def get_set_features_vectorized(feature_data:Dict, max_requests:int=60):
	"""
	Gets features for the final observation.  Vectorized implementation for 
	when the number of final observations > 1.

	Params
	-------------------------
	obs:
		The observation at the end of the booking period. 
	max_requests:
		The number of periods
	"""    
	def get_padded_vector(x, max_requests):
		""" """
		x_padded = np.zeros((x.shape[0], max_requests))
		x_padded[:, :x.shape[1]] = x
		x_padded = torch.FloatTensor(x_padded)

		return x_padded
		
	n = feature_data['n_final_obs']
		
	x_inst = torch.zeros((n, 3))
	x_inst[:, 0] = torch.FloatTensor(feature_data['weight_caps'])
	x_inst[:, 1] = torch.FloatTensor(feature_data['volume_caps'])
	x_inst[:, 2] = feature_data['n_requests']

	x_req = torch.zeros((n, max_requests, 3))
	if feature_data['n_requests'] != 0:
		x_req[:,:,0] = get_padded_vector(feature_data['weights'], max_requests)
		x_req[:,:,1] = get_padded_vector(feature_data['volumes'], max_requests)
		x_req[:,:,2] = get_padded_vector(feature_data['costs'], max_requests)

	x_n_req = torch.ones((n)) * feature_data['n_requests']
	x_n_req = torch.reshape(x_n_req, (-1, 1))

	set_features = {
		'x_inst' : x_inst,
		'x_req' : x_req,
		'x_n_req' : x_n_req 
	}

	return set_features


def formulate_mip_gp(final_observation:Dict):
	"""
	Gets MIP model for the final observation as a Gurobi model.  

	Params
	-------------------------
	final_obs:
		The observation at the end of the booking period. 
	"""

	model = gp.Model()
	model.setParam("OutputFlag", 0)
	model.setParam("Threads", 1)
	model.setObjective(0, sense=gp.GRB.MINIMIZE)

	var_list = []
	constraints = []

	if len(final_observation["requests"]) > 0:
		for i, request in enumerate(final_observation["requests"]):

			var = model.addVar(name=f"x_{i}", vtype="B", obj= - request["cost"])
			var_list.append((var, request))

			var_fixed = model.addVar(name=f"y_{i}", vtype="B", obj=request["cost"], lb=1.0, ub=1.0)
			var_list.append((var_fixed, request))

		model.update()

		weight_constraint = 0
		volume_constraint = 0
		for var, request in var_list:
			if 'x_' in var.varName:
				weight_constraint += request["weight"] * var
				volume_constraint += request["volume"] * var

		constraints.append(model.addConstr(weight_constraint <= final_observation["weight_cap"]))
		constraints.append(model.addConstr(volume_constraint <= final_observation["volume_cap"]))

	return model


def get_reward_from_period_t_to_end(final_obs:Dict, t:int):
	"""	
	Gets the reward from period t to the end of the booking period.  

	Params
	-------------------------
	final_obs:
		The observation at the end of the booking period. 
	t:
		The period to compute the reward from.  
	"""
	total_reward = 0

	for request in final_obs['requests']:
		if request['period'] >= t:
			total_reward += request['price']

	return total_reward



