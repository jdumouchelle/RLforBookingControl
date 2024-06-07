import os
import sys
import subprocess

from typing import List, Dict, Tuple

class VRPSolver(object):

	def __init__(self):
		"""
		Constructor for the VRPSolver class.  This class is implemented as a wrapper for calling the 
		VRP solver provided by Luca et al.
		"""
		pass # do nothing for now


	def set_solver_params(
		self,
		exec_path:str,
		outpath:str,
		parser:str = 'E',                    
		tolerance:float = 0.01,              
		granular_neighbors:int = 25,
		cache:int = 50,
		routemin_iterations:int = 1000,    
		coreopt_iterations:int = 100000,   
		granular_gamma_base:float = 0.250,    
		granular_delta:float = 0.500,     
		shaking_lower_bound:float = 0.375,   
		shaking_upper_bound:float = 0.850,
		seed:int = 0):
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
		self.seed = seed

		self.params = []
		self.params += ["--outpath", f"{self.outpath}"]
		self.params += ["--parser", f"{self.parser}"]
		self.params += ["--tolerance", f"{self.tolerance}"]
		self.params += ["--granular-neighbors", f"{self.granular_neighbors}"]
		self.params += ["--cache", f"{self.cache}"]
		self.params += ["--routemin-iterations", f"{self.routemin_iterations}"]
		self.params += ["--coreopt-iterations", f"{self.coreopt_iterations}"]
		self.params += ["--granular-gamma-base", f"{self.granular_gamma_base}"]
		self.params += ["--granular-delta", f"{self.granular_delta}"]
		self.params += ["--shaking-lower-bound", f"{self.shaking_lower_bound}"]
		self.params += ["--shaking-upper-bound", f"{self.shaking_upper_bound}"]
		self.params += ["--seed", f"{self.seed}"]

		return



	def solve_instance(self, path_to_instance:str):
		"""
		Solves the instance of the specified file path and returns the cost and number of vehicles used.
		
		Params
		-------------------------
		path_to_instance:
			the path the the instance .vrp file.

		Returns
		-------------------------
		a tuple of the operational cost and the number of vehicles used.
		"""

		# execute solver
		res = subprocess.run([self.exec_path, path_to_instance] + self.params)
		assert(res.returncode == 0)

		# recover cost and number of vehicles
		instance_name = path_to_instance.split('/')[-1]
		
		out_file_name = instance_name + '_seed-' + str(self.seed) + '.out'
		sol_file_name = instance_name + '_seed-' + str(self.seed) + '.vrp.sol'

		with open(self.outpath + sol_file_name, 'r') as f:
			sol_data = f.readlines()

		# parser for routes and cost
		routes = []
		with open(self.outpath + sol_file_name, 'r') as f:
			lines = f.readlines()
			for line in lines:
				if 'Route' in line:
					route = line.split(':')[-1]
					route = route[1:].split(' ')
					route = list(map(lambda x: int(x), route))
					routes.append(route)

				else:
					operational_cost = float(line.split(' ')[-1])

		operational_cost = float(sol_data[-1].split(' ')[-1])
		num_vehicles_used = len(routes)

		return operational_cost, num_vehicles_used, routes
