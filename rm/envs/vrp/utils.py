import os
import copy
import torch
import numpy as np


from .vrp_instance_generator import VRPInstanceGenerator
from .product_weights import ProductWeights

from typing import Dict, List, Tuple



#--------------------------------------------------------#
#               	VRP Instances Info 		             #
#--------------------------------------------------------#


def get_weights(cfg):
    """ Gets the weights based on the config.  (Constant for now). """ 
    weight_means = np.array([1])
    weights = ProductWeights(cfg.n_products) 
    weights.set_means(weight_means)
    return weights


def get_prices(cfg):
    """ Gets the prices based on config. """
    prices = {}
    prices[0] = 0

    ### 4 LOCATIONS
    if cfg.n_locations == 4:
        price_at_location =  4 * np.array([0, 1, 2, 3, 4])
        for j in range(1, cfg.n_locations + 1):
            prices[j] = {}
            for i in range(cfg.n_products):
                prices[j][i] = price_at_location[j]

    ### 10 LOCATIONS
    elif cfg.n_locations == 10:
        nodes_grp_1 = (1, 4)
        nodes_grp_2 = (5, 8)
        nodes_grp_3 = (9, 10)

        price_grp_1 = [10]
        price_grp_2 = [12]
        price_grp_3 = [20]

        for j in range(1, cfg.n_locations + 1):
            prices[j] = {}
            for i in range(0, cfg.n_products):
                if j >= nodes_grp_1[0] and j <= nodes_grp_1[1]:
                    prices[j][i] = price_grp_1[i]
                elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1]:
                    prices[j][i] = price_grp_2[i]
                elif j >= nodes_grp_3[0] and j <= nodes_grp_3[1]:
                    prices[j][i] = price_grp_3[i]

    ### 15 LOCATIONS
    elif cfg.n_locations == 15:
        nodes_grp_1 = (1, 5)
        nodes_grp_2 = (6, 10)
        nodes_grp_3 = (11, 15)

        price_grp_1 = [10]
        price_grp_2 = [12]
        price_grp_3 = [20]

        for j in range(1, cfg.n_locations + 1):
            prices[j] = {}
            for i in range(0, cfg.n_products):
                if j >= nodes_grp_1[0] and j <= nodes_grp_1[1]:
                    prices[j][i] = price_grp_1[i]
                elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1]:
                    prices[j][i] = price_grp_2[i]
                elif j >= nodes_grp_3[0] and j <= nodes_grp_3[1]:
                    prices[j][i] = price_grp_3[i]

    ### 50 LOCATIONS
    elif cfg.n_locations == 50:

        nodes_grp_1 = (1, 30)
        nodes_grp_2 = (31, 40)
        nodes_grp_3 = (41, 50)
        
        price_grp_1 = [15]
        price_grp_2 = [22]
        price_grp_3 = [30]
        
        for j in range(1, cfg.n_locations + 1):
            prices[j] = {}
            for i in range(0, cfg.n_products):
                if j >= nodes_grp_1[0] and j <= nodes_grp_1[1]:
                    prices[j][i] = price_grp_1[i]
                elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1]:
                    prices[j][i] = price_grp_2[i]
                elif j >= nodes_grp_3[0] and j <= nodes_grp_3[1]:
                    prices[j][i] = price_grp_3[i]

    ### 100 LOCATIONS
    elif cfg.n_locations == 100:
        nodes_grp_1 = (1, 50)
        nodes_grp_2 = (51, 75)
        nodes_grp_3 = (76, 100)

        price_grp_1 = [15]
        price_grp_2 = [22]
        price_grp_3 = [30]

        for j in range(1, cfg.n_locations + 1):
            prices[j] = {}
            for i in range(0, cfg.n_products):
                if j >= nodes_grp_1[0] and j <= nodes_grp_1[1]:
                    prices[j][i] = price_grp_1[i]
                elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1]:
                    prices[j][i] = price_grp_2[i]
                elif j >= nodes_grp_3[0] and j <= nodes_grp_3[1]:
                    prices[j][i] = price_grp_3[i]
 

    return prices


def get_request_probs(cfg):
    """ Gets the request probabilities based on the config.  """

    probs = {}
    probs[0] = {}

    ### 4 LOCATIONS
    if cfg.n_locations == 4:
        
        inc_dec_rate = 1e-2
        for t in range(cfg.n_periods):
            probs[0][t] = 0.10

        initial_probs = np.array([0.10, 0.45, 0.3, 0.10, 0.05])

        for j in range(1, cfg.n_locations + 1):
            probs[j] = {}
            for i in range(cfg.n_products):
                probs[j][i] = {}    
                for t in range(cfg.n_periods):
                    if t == 0:
                        probs[j][i][t] = initial_probs[j]
                    elif j <= 2:
                        probs[j][i][t] = probs[j][i][t-1] - inc_dec_rate
                    else:
                        probs[j][i][t] = probs[j][i][t-1] + inc_dec_rate

    ### 10 LOCATIONS
    elif cfg.n_locations == 10:
        nodes_grp_1 = (1, 4)
        nodes_grp_2 = (5, 8)
        nodes_grp_3 = (9, 10)

        prob_no_request = 0.10
        total_prob_grp_1 = 0.50
        total_prob_grp_2 = 0.30
        total_prob_grp_3 = 0.10

        prob_grp_1 = total_prob_grp_1 / (nodes_grp_1[1] - nodes_grp_1[0] + 1)
        prob_grp_2 = total_prob_grp_2 / (nodes_grp_2[1] - nodes_grp_2[0] + 1)
        prob_grp_3 = total_prob_grp_3 / (nodes_grp_3[1] - nodes_grp_3[0] + 1)

        initial_probs = {}
        for j in range(1, cfg.n_locations + 1):
            if j >= nodes_grp_1[0] and j <= nodes_grp_1[1] : # in group 1
                 initial_probs[j] = prob_grp_1
            elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1] : # in group 2
                initial_probs[j] = prob_grp_2
            else:
                initial_probs[j] = prob_grp_3

        product_probs = np.array([1.0])
        inc_dec_rate = 1e-3

        for t in range(cfg.n_periods):
            probs[0][t] = prob_no_request            
                        
        for j in range(1, cfg.n_locations + 1):
            probs[j] = {}
            for i in range(cfg.n_products):
                probs[j][i] = {}    
                for t in range(cfg.n_periods):
                    if t == 0:
                        probs[j][i][t] = initial_probs[j]
                    elif j >= nodes_grp_1[0] and j <= nodes_grp_1[1] : # in group 1
                        probs[j][i][t] = probs[j][i][t-1] -  inc_dec_rate
                    elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1] : # in group 2
                        probs[j][i][t] = probs[j][i][t-1] 
                    else:
                        probs[j][i][t] = probs[j][i][t-1] + 2* inc_dec_rate  

    ### 15 LOCATIONS
    elif cfg.n_locations == 15:
        prob_no_request = 0.10
        total_prob_grp_1 = 0.50
        total_prob_grp_2 = 0.30
        total_prob_grp_3 = 0.10

        nodes_grp_1 = (1, 5)
        nodes_grp_2 = (6, 10)
        nodes_grp_3 = (11, 15)

        prob_grp_1 = total_prob_grp_1 / (nodes_grp_1[1] - nodes_grp_1[0] + 1)
        prob_grp_2 = total_prob_grp_2 / (nodes_grp_2[1] - nodes_grp_2[0] + 1)
        prob_grp_3 = total_prob_grp_3 / (nodes_grp_3[1] - nodes_grp_3[0] + 1)

        initial_probs = {}
        for j in range(1, cfg.n_locations + 1):
            if j >= nodes_grp_1[0] and j <= nodes_grp_1[1]: 
                 initial_probs[j] = prob_grp_1
            elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1]:
                initial_probs[j] = prob_grp_2
            else:
                initial_probs[j] = prob_grp_3

        product_probs = np.array([1.0])
        inc_dec_rate = 1e-3

        for t in range(cfg.n_periods):
            probs[0][t] = prob_no_request            
                        
        for j in range(1, cfg.n_locations + 1):
            probs[j] = {}
            for i in range(cfg.n_products):
                probs[j][i] = {}    
                for t in range(cfg.n_periods):
                    if t == 0:
                        probs[j][i][t] = initial_probs[j]
                    elif j >= nodes_grp_1[0] and j <= nodes_grp_1[1]: 
                        probs[j][i][t] = probs[j][i][t-1] -  inc_dec_rate
                    elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1]:
                        probs[j][i][t] = probs[j][i][t-1] 
                    else:
                        probs[j][i][t] = probs[j][i][t-1] +  inc_dec_rate

    ### 50 LOCATIONS
    elif cfg.n_locations == 50:
        prob_no_request = 0.10
        total_prob_grp_1 = 0.50
        total_prob_grp_2 = 0.30
        total_prob_grp_3 = 0.10

        nodes_grp_1 = (1, 30)
        nodes_grp_2 = (31, 40)
        nodes_grp_3 = (41, 50)

        prob_grp_1 = total_prob_grp_1 / (nodes_grp_1[1] - nodes_grp_1[0] + 1)
        prob_grp_2 = total_prob_grp_2 / (nodes_grp_2[1] - nodes_grp_2[0] + 1)
        prob_grp_3 = total_prob_grp_3 / (nodes_grp_3[1] - nodes_grp_3[0] + 1)

        initial_probs = {}
        for j in range(1, cfg.n_locations + 1):
            if j >= nodes_grp_1[0] and j <= nodes_grp_1[1]: 
                 initial_probs[j] = prob_grp_1
            elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1]:
                initial_probs[j] = prob_grp_2
            else:
                initial_probs[j] = prob_grp_3

        product_probs = np.array([1.0])
        inc_dec_rate = 1e-4
        for t in range(cfg.n_periods):
            probs[0][t] = prob_no_request            
                        
        for j in range(1, cfg.n_locations + 1):
            probs[j] = {}
            for i in range(cfg.n_products):
                probs[j][i] = {}    
                for t in range(cfg.n_periods):
                    if t == 0:
                        probs[j][i][t] = initial_probs[j]
                    elif j >= nodes_grp_1[0] and j <= nodes_grp_1[1]: 
                        probs[j][i][t] = probs[j][i][t-1] -  inc_dec_rate
                    elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1]:
                        probs[j][i][t] = probs[j][i][t-1] 
                    else:
                        probs[j][i][t] = probs[j][i][t-1] + 3 * inc_dec_rate
 
    ### 100 LOCATIONS
    elif cfg.n_locations == 100:

        prob_no_request = 0.10
        total_prob_grp_1 = 0.50
        total_prob_grp_2 = 0.30
        total_prob_grp_3 = 0.10

        nodes_grp_1 = (1, 50)
        nodes_grp_2 = (51, 75)
        nodes_grp_3 = (76, 100)

        prob_grp_1 = total_prob_grp_1 / (nodes_grp_1[1] - nodes_grp_1[0] + 1)
        prob_grp_2 = total_prob_grp_2 / (nodes_grp_2[1] - nodes_grp_2[0] + 1)
        prob_grp_3 = total_prob_grp_3 / (nodes_grp_3[1] - nodes_grp_3[0] + 1)

        initial_probs = {}
        for j in range(1, cfg.n_locations + 1):
            if j >= nodes_grp_1[0] and j <= nodes_grp_1[1]: 
                 initial_probs[j] = prob_grp_1
            elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1]:
                initial_probs[j] = prob_grp_2
            else:
                initial_probs[j] = prob_grp_3

        product_probs = np.array([1.0])

        probs = {}
        inc_dec_rate = 1e-5

        probs[0] = {}
        for t in range(cfg.n_periods):
            probs[0][t] = prob_no_request            
                        
        for j in range(1, cfg.n_locations + 1):
            probs[j] = {}
            for i in range(cfg.n_products):
                probs[j][i] = {}    
                for t in range(cfg.n_periods):
                    if t == 0:
                        probs[j][i][t] = initial_probs[j]
                    elif j >= nodes_grp_1[0] and j <= nodes_grp_1[1]: 
                        probs[j][i][t] = probs[j][i][t-1] - 2 * inc_dec_rate
                    elif j >= nodes_grp_2[0] and j <= nodes_grp_2[1]:
                        probs[j][i][t] = probs[j][i][t-1] + 2 * inc_dec_rate
                    else:
                        probs[j][i][t] = probs[j][i][t-1] + 2 * inc_dec_rate 

    return probs


#--------------------------------------------------------#
#               	Get graph-based data                 #
#--------------------------------------------------------#

def get_graph_data(dataset):
    """ Gets set representation for features.  """
    
    def get_inst_features(x):
        dep_x = x['locations'][0][0]
        dep_y = x['locations'][0][1]
        n_nodes = x['features']['num_nodes']
        n_vehciles = x['features']['num_vehicles']
        capacity = x['features']['capacity']    
        return [dep_x, dep_y, n_nodes, n_vehciles, capacity]
    
    def get_loc_features(x):
        loc_features = []
        for i in range(1,  len(x['demands'])):
            if x['demands'][i] == 0:
                continue
            loc_x = x['locations'][i][0]
            loc_y = x['locations'][i][1]
            loc_demand = x['demands'][i]
            loc_features.append([loc_x, loc_y, loc_demand])
        return loc_features
    
    def pad_tensor(t, max_locations):
    
        if t.shape[0] == 0:
            return torch.zeros((max_locations, 3)).float()
        pd = (0, 0, 0, max_locations - t.shape[0])
        p = torch.nn.ZeroPad2d(pd)
        return p(t)
    
    max_locations = len(dataset[0]['locations'])-1
            
    # first-stage decision
    x_inst = np.array(list(map(lambda x: get_inst_features(x), dataset))) 
    x_inst = torch.from_numpy(x_inst).float()

    # targets
    y = None
    if 'operational_cost' in dataset[0]:
	    y = np.array(list(map(lambda x: x['operational_cost'], dataset))).reshape(-1,1)
	    y = torch.from_numpy(y).float()
    
    # scenario padding
    x_loc = list(map(lambda x: np.array(get_loc_features(x)), dataset))
    x_loc = list(map(lambda x: torch.from_numpy(x).float(), x_loc))    
    x_n_loc = np.array(list(map(lambda x: x.shape[0], x_loc))).reshape(-1,1)
    x_n_loc = torch.from_numpy(x_n_loc).float()
    x_loc = list(map(lambda x: pad_tensor(x, max_locations), x_loc))
    x_loc = torch.stack(x_loc)
    
    return x_inst, x_loc, x_n_loc, y



#--------------------------------------------------------#
#               	Feature Extraction                   #
#--------------------------------------------------------#

def compute_linear_vrp_features(
	vrp_instance:"VRPInstanceGenerator",
	demands:np.ndarray):
	"""
	Computes a set of linear features for a VRP instance.
	
	Params
	-------------------------
	vrp_instance:
		An instance of VRPInstanceGenreator which has already been initialized.
	demands:	
		An array of the demands at each location. 
	"""

	features = {}

	# get depot location
	depot_x, depot_y = vrp_instance.depot_location

	# get all non-zero demands
	locations_with_demand = []
	location_demands = []
	for j in range(vrp_instance.num_locations+1):
		if demands[j] > 0:
			locations_with_demand.append(vrp_instance.delivery_locations[j-1])
			location_demands.append(demands[j])

	assert(len(locations_with_demand) == (demands != 0).sum()) # check all locations counted

	# compute pairwise-distance between locations (not including depot)
	distance_matrix = np.zeros((len(locations_with_demand), len(locations_with_demand)))
	for i in range(0, len(locations_with_demand)): # changed
		for j in range(i, len(locations_with_demand)):

			# fill diagonal with NaN
			if i == j: 
				distance_matrix[i][j] = np.nan
				continue

			d_ji = np.linalg.norm(np.array(locations_with_demand[i]) - np.array(locations_with_demand[j]))
			distance_matrix[i][j] = d_ji
			distance_matrix[j][i] = d_ji

	# compute distance between depot and locations
	distance_depot = np.zeros(len(locations_with_demand))
	for i in range(0, len(locations_with_demand)):
		distance_depot[i] = np.linalg.norm(np.array(depot_x, depot_y) - np.array(locations_with_demand[i]))

	# depot location
	features['depot_x'] = depot_x
	features['depot_y'] = depot_y

	# average location features
	if len(locations_with_demand) != 0:
		features['min_x'] = np.min(list(map(lambda x: x[0], locations_with_demand)))
		features['max_x'] = np.max(list(map(lambda x: x[0], locations_with_demand)))
		features['mean_x'] = np.mean(list(map(lambda x: x[0], locations_with_demand)))
		features['median_x'] = np.median(list(map(lambda x: x[0], locations_with_demand)))
		features['std_x'] = np.std(list(map(lambda x: x[0], locations_with_demand)))
		features['q25_x'] = np.quantile(list(map(lambda x: x[0], locations_with_demand)), 0.25)
		features['q75_x'] = np.quantile(list(map(lambda x: x[0], locations_with_demand)), 0.75)

		features['min_y'] = np.min(list(map(lambda x: x[1], locations_with_demand)))
		features['max_y'] = np.max(list(map(lambda x: x[1], locations_with_demand)))
		features['mean_y'] = np.mean(list(map(lambda x: x[1], locations_with_demand)))
		features['median_y'] = np.median(list(map(lambda x: x[1], locations_with_demand)))
		features['std_y'] = np.std(list(map(lambda x: x[1], locations_with_demand)))
		features['q25_y'] = np.quantile(list(map(lambda x: x[1], locations_with_demand)), 0.25)
		features['q75_y'] = np.quantile(list(map(lambda x: x[1], locations_with_demand)), 0.75)

	else:
		features['min_x'] = 0
		features['max_x'] = 0
		features['mean_x'] = 0
		features['median_x'] = 0
		features['std_x'] = 0
		features['q25_x'] = 0
		features['q75_x'] = 0

		features['min_y'] = 0
		features['max_y'] = 0
		features['mean_y'] = 0
		features['median_y'] = 0
		features['std_y'] = 0
		features['q25_y'] = 0
		features['q75_y'] = 0

	# average demand features
	if len(locations_with_demand) != 0:
		features['min_demand'] = np.min(location_demands)
		features['max_demand'] = np.max(location_demands)
		features['mean_demand'] = np.mean(location_demands)
		features['median_demand'] = np.median(location_demands)
		features['std_demand'] = np.std(location_demands)
		features['q25_demand'] = np.quantile(location_demands, 0.25)
		features['q75_demand'] = np.quantile(location_demands, 0.75)

	else:
		features['min_demand'] = 0
		features['max_demand'] = 0
		features['mean_demand'] = 0
		features['median_demand'] = 0
		features['std_demand'] = 0
		features['q25_demand'] = 0
		features['q75_demand'] = 0


	# vehicle features
	features['capacity'] = int(vrp_instance.capacity)
	features['num_vehicles'] = vrp_instance.num_vehicles

	# depot distance features
	if len(locations_with_demand) != 0:
		features['min_dist_depot'] = np.min(distance_depot)
		features['max_dist_depot'] = np.max(distance_depot)
		features['mean_dist_depot'] = np.mean(distance_depot)
		features['median_dist_depot'] = np.median(distance_depot)
		features['std_dist_depot'] = np.std(distance_depot)
		features['q25_dist_depot'] = np.quantile(distance_depot, 0.25)
		features['q75_dist_depot'] = np.quantile(distance_depot, 0.75)

	else:
		features['min_dist_depot'] = 0
		features['max_dist_depot'] = 0
		features['mean_dist_depot'] = 0
		features['median_dist_depot'] = 0
		features['std_dist_depot'] = 0
		features['q25_dist_depot'] = 0
		features['q75_dist_depot'] = 0


	assert(not np.isnan(features['min_dist_depot']))
	assert(not np.isnan(features['max_dist_depot'] ))
	assert(not np.isnan(features['mean_dist_depot'] ))
	assert(not np.isnan(features['median_dist_depot'] ))
	assert(not np.isnan(features['std_dist_depot'] ))
	assert(not np.isnan(features['q25_dist_depot'] ))
	assert(not np.isnan(features['q75_dist_depot'] ))

	# relative location feaures
	if distance_matrix.shape[0] > 1:
		features['min_dist'] = np.nanmin(distance_matrix)
		features['max_dist'] = np.nanmax(distance_matrix)
		features['mean_dist'] = np.nanmean(distance_matrix)
		features['median_dist'] = np.nanmedian(distance_matrix)
		features['std_dist'] = np.nanstd(distance_matrix)
		features['q25_dist'] = np.nanquantile(distance_matrix, 0.25)
		features['q75_dist'] = np.nanquantile(distance_matrix, 0.75)

	else: # case where only one entry exists and all quantites will be NaN
		features['min_dist'] = 0
		features['max_dist'] = 0
		features['mean_dist'] = 0
		features['median_dist'] = 0
		features['std_dist'] = 0
		features['q25_dist'] = 0
		features['q75_dist'] = 0

	# number of node features
	features['num_nodes'] = len(locations_with_demand)

	return features


#--------------------------------------------------------#
#               	Instance Writing                     #
#--------------------------------------------------------#

def write_vrp_instance(
	instance_path:str,
	instance_name:str, 
	comment:str,
	capacity:float,
	depot_location:np.ndarray, 
	delivery_locations:List[np.ndarray],
	demands:np.ndarray):
	"""
	Writes the vrp instance to a file, which can then be solved by filo.  This function does not include the 
	offset.  
	
	Params
	-------------------------
	instance_path: 
		The path where the instance should be saved.
	instance_name:
		The name of the instance.
	comment:
		A comment for the instance.  
	capacity:
		The capacity of each vehicles.
	depot_location:
		The depot location.
	delivery_locations:
		A list of the delivery locations.
	demands:
		An array of the demands at each location.  
	"""

	vrp_type = 'CVRP'
	dimension = (demands != 0).sum() + 1 # add 1 for depot
	edge_weight_type = 'EUC_2D'
	capacity = int(capacity)
	num_locations = len(delivery_locations) #+ 1

	vrp_instance_str = ''

	# write problem information
	vrp_instance_str += 'NAME : ' + instance_name + '\n'
	vrp_instance_str += 'COMMENT : (' + comment + ')\n'
	vrp_instance_str += 'TYPE : ' + vrp_type + '\n'
	vrp_instance_str += 'DIMENSION : ' + str(dimension) + '\n'
	vrp_instance_str += 'EDGE_WEIGHT_TYPE : ' + edge_weight_type + '\n'
	vrp_instance_str += 'CAPACITY : ' + str(capacity) + '\n'

	# write coordinates for delivery locations
	vrp_instance_str += 'NODE_COORD_SECTION \n'
	depot_x, depot_y = depot_location
	vrp_instance_str += ' ' + str(1) + ' ' + str(depot_x) + ' ' + str(depot_y) + '\n' # add depot

	idx = 2
	for j in range(num_locations):
		if demands[j+1] != 0:
			x, y = delivery_locations[j]
			vrp_instance_str += ' ' + str(idx) + ' ' + str(x) + ' ' + str(y) + '\n' 
			idx += 1

	# write demands for delivery locations
	vrp_instance_str += 'DEMAND_SECTION \n'
	vrp_instance_str += str(1) + ' 0\n' # add depot

	idx = 2
	for j in range(num_locations):
		if demands[j+1] != 0:
			vrp_instance_str += str(idx) + ' ' + str(int(demands[j+1])) + '\n'    
			idx += 1 

	# depot section
	vrp_instance_str += 'DEPOT_SECTION \n'
	vrp_instance_str += ' 1 \n'
	vrp_instance_str += ' -1 \n'
	vrp_instance_str += 'EOF \n'

	with open(instance_path, 'w') as f:
		f.write(vrp_instance_str)

	return


def write_vrp_instance_offset(
	instance_path:str,
	instance_name:str, 
	comment:str,
	capacity:float,
	depot_location:np.ndarray, 
	delivery_locations:List[np.ndarray],
	demands:np.ndarray):
	"""
	Writes the vrp instance to a file, which can then be solved by filo.  This function includes the offset to 
	the depot location.  
	
	Params
	-------------------------
	instance_path: 
		The path where the instance should be saved.
	instance_name:
		The name of the instance.
	comment:
		A comment for the instance.  
	capacity:
		The capacity of each vehicles.
	depot_location:
		The depot location.
	delivery_locations:
		A list of the delivery locations.
	demands:
		An array of the demands at each location.  
	"""
	vrp_type = 'CVRP'
	dimension = (demands != 0).sum() + 1 # add 1 for depot
	edge_weight_type = 'EXPLICIT'
	capacity = int(capacity)
	num_locations = len(delivery_locations) #+ 1

	vrp_instance_str = ''

	# write problem information
	vrp_instance_str += 'NAME : ' + instance_name + '\n'
	vrp_instance_str += 'COMMENT : (' + comment + ')\n'
	vrp_instance_str += 'TYPE : ' + vrp_type + '\n'
	vrp_instance_str += 'DIMENSION : ' + str(dimension) + '\n'
	vrp_instance_str += 'EDGE_WEIGHT_TYPE : ' + edge_weight_type + '\n'
	vrp_instance_str += 'CAPACITY : ' + str(capacity) + '\n'

	# write distances for delivery locations

	# get actual set of locations
	delivery_locations_with_demand = [copy.deepcopy(depot_location)]

	for j in range(num_locations):
		if demands[j+1] != 0:
			delivery_locations_with_demand.append(copy.deepcopy(delivery_locations[j]))

	distance_matrix = np.zeros((len(delivery_locations_with_demand), len(delivery_locations_with_demand)))
	for j in range(len(delivery_locations_with_demand)):
		for i in range(len(delivery_locations_with_demand)):
			distance_matrix[j][i] = np.linalg.norm(delivery_locations_with_demand[j] - delivery_locations_with_demand[i])

	depot_offset = 100

	# write distances for node coord section
	vrp_instance_str += 'NODE_COORD_SECTION \n'
	for j in range(0, len(delivery_locations_with_demand)):
		for i in range(j+1, len(delivery_locations_with_demand)):
			if j == 0:
				vrp_instance_str += f"{j+1} {i+1} {distance_matrix[i][j] + depot_offset} \n"
			else:
				vrp_instance_str += f"{j+1} {i+1} {distance_matrix[i][j]} \n"

	# write demands for delivery locations
	vrp_instance_str += 'DEMAND_SECTION \n'
	vrp_instance_str += str(1) + ' 0\n' # add depot

	idx = 2
	for j in range(num_locations):
		if demands[j+1] != 0:
			vrp_instance_str += str(idx) + ' ' + str(int(demands[j+1])) + '\n'    
			idx += 1 

	# depot section
	vrp_instance_str += 'DEPOT_SECTION \n'
	vrp_instance_str += ' 1 \n'
	vrp_instance_str += ' -1 \n'
	vrp_instance_str += 'EOF \n'

	with open(instance_path, 'w') as f:
		f.write(vrp_instance_str)

	return


#--------------------------------------------------------#
#			 	State storing functions                  #
#--------------------------------------------------------#

def get_state_as_tuple(state, num_locations:int, num_products:int):
	"""
	Gets the states as a tuple.

	Params
	-------------------------
	state: 
		The state as a tuple, dict, or matrix.  
	num_locations:
		The number of locations
	num_products:
		The number of products.
	"""
	if isinstance(state, tuple):
		return state

	# convert matrix to tuple
	elif isinstance(state, np.ndarray):
		state_as_tuple = tuple(state.reshape(-1))

	# convert dict to tuple
	else:
		state_as_list = []
		for j in range(1, num_locations+1):
			for i in range(num_products):
				state_as_list.append(state[j][i])
		state_as_tuple = tuple(state_as_list)

	return state_as_tuple
	

def get_state_as_matrix(state, num_locations:int, num_products:int):
	"""
	Gets the state as a matrix.

	Params
	-------------------------
	state: 
		The state as a tuple, dict, or matrix.  
	num_locations:
		The number of locations
	num_products:
		The number of products.

	"""
	if isinstance(state, np.ndarray):
		return state

	# convert dict to tuple if needed
	if isinstance(state, dict):
		state = get_state_as_tuple(state, num_locations, num_products)

	# convert tuple to matrix
	state_as_matrix = np.zeros((num_locations, num_products))
	idx = 0
	for j in range(num_locations):
		for i in range(num_products):
			
			state_as_matrix[j][i] = state[idx]
			idx += 1
			
	return state_as_matrix



def get_state_as_dict(state, num_locations:int, num_products:int):
	"""
	Gets the state as a dictionary.

	Params
	-------------------------
	state: 
		The state as a tuple, dict, or matrix.  
	num_locations:
		The number of locations
	num_products:
		The number of products.
	"""
	if isinstance(state, dict):
		return state

	if isinstance(state, np.ndarray):
		state = get_state_as_tuple(state, num_locations, num_products)
		
	state_as_dict = {}
	idx = 0
	for j in range(1, num_locations+1):
		state_as_dict[j] = {}
		for i in range(num_products):
			state_as_dict[j][i] = state[idx]
			idx += 1
		
	return state_as_dict
