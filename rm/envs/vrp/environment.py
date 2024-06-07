import numpy as np
import copy

from .product_weights import ProductWeights
from .observation import Observation
from .vrp_instance_generator import VRPInstanceGenerator
from .booking_period_env import BookingPeriodEnv
from .vrp_solver import VRPSolver

from .utils import *

from typing import Dict, List, Tuple


class Environment(object):

	def __init__(self):
		"""
		Constuctor for Environment .  Does nothing.
		"""
		pass

	def seed_env(self, seed:int):
		"""
		Seeds the booking period for reproducibility.

		Params
		-------------------------
		seed: 
			An integer which seeds the Environment . 
		"""
		self.env.seed(seed)

	def set_booking_period_parameters(self, 
		num_locations:int, 
		num_products:int, 
		num_periods:int, 
		probs:Dict, 
		prices:Dict, 
		weights:"ProductWeights"):
		"""
		Sets parameters for the booking period.  

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

		# check total probability sums to 1
		self._check_total_probability()


	def set_vrp_parameters(self, 
		num_vehicles:int, 
		load_factor:float, 
		added_vehicle_cost:float, 
		location_bounds:Tuple[float]):
		"""
		Sets parameters of vehicle routing problem.
		
		Params
		-------------------------
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
		self.load_factor = load_factor
		self.added_vehicle_cost = added_vehicle_cost
		self.location_bounds = location_bounds


	def set_solver_parameters(self, 
		exec_path:str,
		outpath:str,
		parser:str,                  
		tolerance:float,             
		granular_neighbors:int,
		cache:int,
		routemin_iterations:int, 
		coreopt_iterations:int,
		granular_gamma_base:float, 
		granular_delta:float, 
		shaking_lower_bound:float, 
		shaking_upper_bound:float,
		seed:int):
		"""
		Sets the parameters of filo, the VRP solver used for this implementation. 

		For details on parameters see filo documentation [ADD WHEN AVALIBLE].  

		Params
		-------------------------
		exec_path:
			path to the filo executable file
		outpath:
			Output directory.
		parser:
			Parser type, it can be X, K, Z, or E.
		tolerance:
			Floating point tolerance.
		granular-neighbors:
			Neighbors per vertex in granular neighborhoods.
		cache:
			Selective cache dimension.
		routemin-iterations:
			Max route minimization iterations.
		coreopt-iterations:
			Core optimization iterations.
		granular-gamma-base: 
			Initial sparsification factor gamma base.
		granular-delta:
			Granular reduction factor delta.
		shaking-lower-bound:
			Shaking lower bound factor.
		shaking-upper-bound:
			Shaking upper bound factor.
		seed:                      
			Seed for the solver.
		"""
		self.exec_path = exec_path
		self.outpath = outpath
		self.parser = parser                 
		self.tolerance = tolerance            
		self.granular_neighbors = granular_neighbors
		self.cache = cache
		self.routemin_iterations = routemin_iterations  
		self.coreopt_iterations = coreopt_iterations
		self.granular_gamma_base = granular_gamma_base 
		self.granular_delta = granular_delta  
		self.shaking_lower_bound = shaking_lower_bound
		self.shaking_upper_bound = shaking_upper_bound
		self.solver_seed = seed


	def initialize(self, seed:int = 0, file_exec_path_prefix='./'):
		"""
		Initializes the booking period environment , vrp instance, vrp solver.
		The parameters are set according to what is passed into set_booking_period_parameters,
		set_vrp_parameters, and set_solver_parameters.

		Params
		-------------------------
		seed:
			The seed which will be used to seed the vrp instance and solver.
		"""

		# initialize booking period enviornemnt
		self.env = BookingPeriodEnv(
			num_locations = self.num_locations,
			num_products = self.num_products,
			num_periods = self.num_periods,
			probs = self.probs, 
			prices = self.prices,
			weights = self.weights)

		self.seed = self.env.seed
		self.step = self.env.step
		self.reset = self.env.reset

		# initialize vrp instance
		self.vrp = VRPInstanceGenerator(
			num_locations = self.num_locations,
			num_products = self.num_products,
			num_periods = self.num_periods,
			probs = self.probs, 
			weights = self.weights, 
			num_vehicles = self.num_vehicles, 
			load_factor = self.load_factor, 
			added_vehicle_cost = self.added_vehicle_cost,
			location_bounds = self.location_bounds)

		self.vrp.seed(seed)
		self.vrp.generate_random()
		self.vrp.compute_capacity()

		# add all vrp parameters
		self.expected_demands = self.vrp.expected_demands
		self.capacity = self.vrp.capacity
		self.locations = self.vrp.locations
		self.depot_location = self.vrp.depot_location
		self.delivery_locations = self.vrp.delivery_locations

		# initialize solver
		self.solver = VRPSolver()
		self.solver.set_solver_params(
			exec_path = self.exec_path,
			outpath = self.outpath,
			parser = self.parser,                    
			tolerance = self.tolerance,              
			granular_neighbors = self.granular_neighbors,
			cache = self.cache,
			routemin_iterations = self.routemin_iterations,    
			coreopt_iterations = self.coreopt_iterations,   
			granular_gamma_base = self.granular_gamma_base,    
			granular_delta = self.granular_delta,     
			shaking_lower_bound = self.shaking_lower_bound,   
			shaking_upper_bound = self.shaking_upper_bound,
			seed = self.solver_seed)

		return

	
	def compute_approximate_cost(self, obs:"Observation", model:"MLValueEstimator"):
		"""
		Computes the approximate vrp cost using the ML value estimator provided.  

		Params
		-------------------------
		obs:
			an observation which the operational cost will be computed with respect to.
		model:
			the model used to predict operational cost
		"""

		# get features
		demands, weights_as_requests = self.get_demands_at_obs(obs)
		weights_bpp = self.get_bpp_weights(weights_as_requests)

		features = compute_linear_vrp_features(self, demands)

		vrp_info = {
			'obs' : obs,
			'demands' : demands,
			'features' : features,
			'weights' : weights_as_requests,
			'weights_bpp' : weights_bpp,
			'capacity' : self.capacity,
			'locations' : self.locations,
		}

		return model.predict([vrp_info])[0]


	def compute_exact_cost(self, obs:"Observation", instance_path:str):
		"""
		Computes the exact vrp cost for an observation.  

		Params
		-------------------------
		obs:
			an observation which the operational cost will be computed with respect to.
		instance_path:
			a path in which the instance file will be saved to (both the .vrp and .log)
			files will be saved here.  
		"""
		results = self.compute_vrp_cost(obs, instance_path)

		return results['adjusted_operational_cost']


	def compute_vrp_cost(self, obs:"Observation", instance_path:str):
		"""
		Computes the vrp cost for an observation.  Results are returned as a dictionary containing
		all information regarding the instance.  

		Params
		-------------------------
		obs:
			an observation which the operational cost will be computed with respect to.
		instance_path:
			a path in which the instance file will be saved to (both the .vrp and .log)
			files will be saved here.  
		"""
		def write_vrp_problem(
			capacity:float, 
			depot_location:np.ndarray, 
			delivery_locations:List[np.ndarray], 
			demands:np.ndarray, 
			instance_path:str):
			"""
			Writes the vrp instances to a .vrp files.  This is just a wrapper for the utils.py function.
			The split demand locations should be passed here.  

			Params
			-------------------------
			capacity:
				The capacity for each vehicle.  
			depot_location:
				The depot location.
			delivery_location:
				A list of the delivery locations.
			demands:
				An array of the demands.
			instance_path:
				Path to save the instance to.
			"""
			instance_name = instance_path.split('/')[-1]
			comment = 'n/a'

			write_vrp_instance_offset(
				instance_path = instance_path,
				instance_name = instance_name, 
				comment = comment,
				capacity = capacity,
				depot_location = depot_location, 
				delivery_locations = delivery_locations,
				demands = demands)

			return

		def write_vrp_log(
			capacity:float, 
			split_depot_location:np.ndarray, 
			split_delivery_locations:List[np.ndarray], 
			split_demands:np.ndarray, 
			depot_location:np.ndarray, 
			delivery_locations:List[np.ndarray], 
			demands:np.ndarray, 
			instance_path:str):
			"""
			Writes the vrp instances to a .log files.  This is mainly used for 
			debugging and verifying that the split locations are accurate.  

			Params
			-------------------------
			capacity:
				The capacity for each vehicle.  
			split_depot_location:
				The split depot location.
			split_delivery_location:
				A list of the split delivery locations.
			split_demands:
				An array of the split demands.
			depot_location:
				The depot location.
			delivery_location:
				A list of the delivery locations.
			demands:
				An array of the demands.
			instance_path:
				Path to save the instance to.
			"""

			log_path = instance_path.replace('.vrp', '.log')

			with open(log_path, 'w') as f:
				f.write(f"Capacity: {capacity}\n")

				# split demands/locations
				f.write(f"Split Depot Location: {split_depot_location}\n")
				f.write("Split Delivery Locations:\n")
				for location in split_delivery_locations:
					f.write(f"  {location}\n")
				f.write(f'Split Demands: {split_demands}\n')

				# Original demands/locations
				f.write(f"Depot Location: {depot_location}\n")
				f.write("Delivery Locations:\n")
				for location in delivery_locations:
					f.write(f"  {location}\n")
				f.write(f'Demands: {demands}\n')

			return
		
		# start of compute_vrp_cost
		demands, weight_requests = self.get_demands_at_obs(obs)
		
		# if no locations
		if (demands != 0).sum() == 0:
			operational_cost = 0
			num_vehicles_used = 0
			adjusted_operational_cost = 0
			filo_cost = 0
			weights_bpp = []
			routes = []
			int_fp_cost = 0
			
		else:
			# get split demands and bin packing weights
			split_demands, split_locations, split_location_dict = self._vrp_to_split_cost(weight_requests)
			weights_bpp = self.get_bpp_weights(weight_requests)

			write_vrp_problem(
				capacity = self.capacity, 
				depot_location = split_locations[0], 
				delivery_locations = split_locations[1:], 
				demands = split_demands, 
				instance_path = instance_path)

			write_vrp_log(
				capacity = self.capacity, 
				split_depot_location = split_locations[0], 
				split_delivery_locations = split_locations[1:], 
				split_demands = split_demands, 
				depot_location = self.locations[0],
				delivery_locations = self.locations[1:],
				demands = demands,
				instance_path = instance_path)

			filo_cost, num_vehicles_used, routes_split = self.solver.solve_instance(instance_path)

			# Map split routes to actual routes
			routes = []
			for route_split in routes_split:
				route = list(map(lambda x: split_location_dict[x], route_split))
				routes.append(route)

			# compute operational cost as floating point
			int_fp_cost = 0
			operational_cost = 0
			for route in routes:
				int_fp_cost += np.round(self.vrp.distance_matrix[0][route[0]]) #+ offset
				operational_cost += self.vrp.distance_matrix[0][route[0]]

				if len(route) != 1:
					for i in range(len(route)-1):
						int_fp_cost += np.round(self.vrp.distance_matrix[route[i]][route[i+1]])
						operational_cost += self.vrp.distance_matrix[route[i]][route[i+1]]

				int_fp_cost += np.round(self.vrp.distance_matrix[route[-1]][0]) #+ offset
				operational_cost += self.vrp.distance_matrix[route[-1]][0]

			if num_vehicles_used > self.vrp.num_vehicles:
				adjusted_operational_cost = operational_cost + self.added_vehicle_cost * (num_vehicles_used - self.num_vehicles)
			else:
				adjusted_operational_cost = operational_cost

		# get vrp features for linear model
		features = compute_linear_vrp_features(self.vrp, demands)
		
		results = {
			'obs' : obs,
			'adjusted_operational_cost' : adjusted_operational_cost,
			'num_vehicles_used' : num_vehicles_used,
			'operational_cost' : operational_cost,
			'rounded_operational_cost' : int_fp_cost,
			'filo_cost' : filo_cost,
			'demands' : demands,
			'routes' : routes,
			'weights' : weight_requests,
			'weights_bpp' : weights_bpp,
			'locations' : self.locations,
			'capacity' : self.capacity,
			'features' : features,
		}
		
		return results


	def get_demands_at_obs(self, obs_or_state:"Observation"):
		"""
		Samples demands and requests at the end of the decision making phase.

		Params
		-------------------------
		obs:
			an observation which the which is used to given requests which are sample from.  
		"""
		demands = np.zeros(self.num_locations + 1)

		weight_requests = self.get_weight_demands_at_location(obs_or_state)

		for j in range(1, self.num_locations + 1):
			demands[j] = np.sum(weight_requests[j])
			
		return demands, weight_requests

	def get_weight_demands_at_location(self, obs_or_state:"Observation"):
		"""
		Samples weights of requests given by the input observation.

		Params
		-------------------------
		obs:
			an observation which the which is used to given requests which are sample from. 
		"""

		# get obs_or_state as dictionary
		if isinstance(obs_or_state, Observation):
			state = obs_or_state.state_dict

		elif isinstance(obs_or_state, tuple) or isinstance(obs_or_state, np.ndarray):
			state = get_state_as_dict(obs_or_state, self.num_locations, self.num_products)
		elif isinstance(obs_or_state, dict):
			state = obs_or_state
		else:
			raise Exception('State passed must be of type dict, tuple, np.ndarray, or observation.')

		weight_requests = {}
		weight_requests[0] = []

		for j in range(1, self.num_locations + 1):
			weight_requests[j] = []
			for i in range(self.num_products):
				w_ji = state[j][i]
				for k in range(int(w_ji)):
					weight_requests[j].append(self.weights.sample(i))
			
		return weight_requests


	def get_remaining_expected_demands(self, period:int):
		"""
		Gets the remaining expected demand from period 'period' onward.  This method is 
		only required for the booking limit policy provided from GGM.  

		Params
		-------------------------
		period:
			The period in which the remaining expected demand will be computed from.  
		"""

		expected_demands = np.zeros(self.num_locations+1)

		for j in range(1, self.num_locations + 1):
			for i in range(0, self.num_products):
				for t in range(period, self.num_periods):
					prob_jit = self.probs[j][i][t]
					w_i = self.weights.get_mean(i)
					expected_demands[j] += prob_jit * w_i

		return expected_demands



	def get_bpp_weights(self, weight_requests:Dict):
		"""
		Computes the weights which will be used in the bin packing problem.  The bin packing weights
		will only be used in the case of when the operational cost predictor uses a bpp solver to 
		adjust for outsourcing vehicles.  

		Params
		-------------------------
		weight_requests:
			A dictionary contianing the weight requests at each location.  

		"""
		weights_bpp = []

		for j in range(1, self.num_locations+1):

			# if no demand at location, then skip
			if len(weight_requests[j]) == 0:
				continue

			# determine number of vehicles needed to meet demand at location
			curr_bin = []
			for weight in weight_requests[j]:
				if np.sum(curr_bin) + weight >= self.capacity:
					weights_bpp.append(np.sum(curr_bin))
					curr_bin = [weight]
				else:
					curr_bin.append(weight)
			weights_bpp.append(np.sum(curr_bin))	

		return weights_bpp


	def _vrp_to_split_cost(self, weight_requests:Dict):
		"""
		Formulates the split cost vrp problem by simply packing grouping items
		until the capacity is reached.  

		Params
		-------------------------
		weight_requests:
			A dictionary contianing the weight requests at each location.  

		"""
		vrp_ = copy.deepcopy(self.vrp)

		locations = [copy.deepcopy(self.vrp.locations[0])]
		demands = [0]
		split_location_dict = {}
		loc_index = 1

		for j in range(1, self.num_locations+1):

			# if no demand at location, then skip
			if len(weight_requests[j]) == 0:
				demands.append(0)
				locations.append(copy.deepcopy(self.vrp.locations[j]))
				continue

			# determine number of vehicles needed to meet demand at location
			bins = []
			curr_bin = []
			n_bins = 1
			for weight in weight_requests[j]:

				if np.sum(curr_bin) + weight >= self.capacity:
					bins.append(curr_bin)
					curr_bin = [weight]
					n_bins += 1

				else:
					curr_bin.append(weight)
			bins.append(curr_bin)		

			# duplication locations to meet requests
			for i in range(n_bins):
				demands.append(np.sum(bins[i]))
				locations.append(copy.deepcopy(self.vrp.locations[j]))
				split_location_dict[loc_index] = j
				loc_index += 1

		demands = np.array(demands)

		return demands, locations, split_location_dict

	def _check_total_probability(self):
		"""
			Asserts that the total probability of the probs given is 1 at each time step.  
		"""
	
		for t in range(self.num_periods):
			total_prob = 0 
			for j in range(0, self.num_locations+1):
				if j == 0:
					total_prob += self.probs[0][t]
					continue
				for i in range(0, self.num_products):
					total_prob += self.probs[j][i][t]
					
			assert(np.abs(total_prob-1) < 1e-5)
		return 


	def copy(self):
		return copy.deepcopy(self)