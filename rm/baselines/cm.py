import time
import pickle
import copy 

import numpy as np
import gurobipy as gp

from collections import defaultdict

from rm.envs.cm.utils import *
from rm.envs.cm.environment import Environment


#--------------------------------------------------------#
#               First come first serve                   #
#--------------------------------------------------------#

class FCFS(object):
	
	
	def __init__(self, env):
		
		self.env = env
		
	def get_action(self, obs):
		
		if len(obs.action_set) == 1:
			return 0
		
		action = 1
		used_weight, used_volume = self.get_utilized_weight_volume()
		if self.env.weights[obs.request[0]] + used_weight > self.env.expected_weight_capacity:
			action =  0
		elif self.env.volumes[obs.request[0]] + used_volume > self.env.expected_volume_capacity:
			action = 0
			
		return action
		
	def get_utilized_weight_volume(self):
		
		# get counts for each item
		item_count = np.zeros(self.env.n_items)
		for key, value in self.env.state.items():
			item_count[key[0]] += value
			
		used_weight = np.array(self.env.weights) @ np.array(item_count)
		used_volume = np.array(self.env.volumes) @ np.array(item_count)
		
		return used_weight, used_volume




#--------------------------------------------------------#
#         Deterministic Linear Programming (DLP)         #
#--------------------------------------------------------#

class DLP(object):

    def __init__(self, env):
        self.env = env

        self.n_classes = self.env.n_items
        self.n_price_ratios = self.env.n_price_ratios
        self.n_items = self.n_classes * self.n_price_ratios
        self.num_periods = self.env.num_periods


        self.d = self._get_expected_demand()
        self.f = self._get_expected_price()
        self.w = self._get_expected_chargeable_weight()


    def init(self):
        self.get_b()
        return

    def get_action(self, obs):

        if len(obs.action_set) == 1:
            return 0

        # set action to 1 if oppurtunity cost is exceeded
        request_price_dict = {0.7 : 0, 1.0 : 1, 1.4 : 2} 
        item = obs.request[0] + self.n_classes * request_price_dict[obs.request[1]]
        idx = item % self.n_classes
        price = self.f[item] * self.w[item]
        cost = self.env.weights[idx] * self.b_w + self.env.volumes[idx] * self.b_v

        action = 0
        if price >= cost:
            action = 1

        used_weight, used_volume = self.get_utilized_weight_volume()
        if self.env.weights[idx] + used_weight > self.env.expected_weight_capacity:
            action =  0
        elif self.env.volumes[idx] + used_volume > self.env.expected_volume_capacity:
            action = 0

        return action


    def get_utilized_weight_volume(self):

        # get counts for each item
        item_count = np.zeros(self.env.n_items)
        for key, value in self.env.state.items():
            item_count[key[0]] += value

        used_weight = np.array(self.env.weights) @ np.array(item_count)
        used_volume = np.array(self.env.volumes) @ np.array(item_count)

        return used_weight, used_volume


    def get_b(self):
        model = self.get_lp()

        model.optimize()

        # recover dual values for constraints
        weight_cons = model.getConstrByName("weight_cons")
        volume_cons = model.getConstrByName("volume_cons")

        self.b_w = weight_cons.Pi
        self.b_v = volume_cons.Pi

        return

    def get_lp(self):
        return self._formulate_lp(self.d)

    def _formulate_lp(self, d):
        """ Formulates the lp. """
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        model.setObjective(0, sense=gp.GRB.MAXIMIZE)
        
        # variables for y, z
        y = []
        z = []
        for i in range(self.n_items):
            y.append(model.addVar(name="y_{i}", obj = self.w[i] * self.f[i]))
            z.append(model.addVar(name="y_{i}", obj = - self.env.cost_multiplier))

        # variables for x
        x_w = []
        x_v = []
        for i in range(self.n_items):
            x_w.append(model.addVar(name=f'x_w_{i}', lb= - gp.GRB.INFINITY))
            x_v.append(model.addVar(name=f'x_v_{i}', lb= - gp.GRB.INFINITY))

        # demand constraint (20)
        for i in range(self.n_items):
            model.addConstr(y[i] <= d[i])

        # weight constraint (21)
        eq_ = 0
        for x_w_i in x_w:
            eq_ += x_w_i
        model.addConstr(eq_ <= self.env.expected_weight_capacity, name="weight_cons")

        # volume constraint (22)
        eq_ = 0
        for x_v_i in x_v:
            eq_ += x_v_i
        model.addConstr(eq_ <= self.env.expected_volume_capacity, name="volume_cons")

        # weight y,z constraint (23)
        for i in range(self.n_items):
            idx = i % self.n_classes
            model.addConstr(y[i] * self.env.weights[idx] - z[i] - x_w[i] <= 0, name=f"weight_{i}_cons" )

        # volume y,z constraint (24)
        for i in range(self.n_items):
            idx = i % self.n_classes
            model.addConstr(y[i] * self.env.volumes[idx] / self.env.weight_volume_ratio - z[i] - x_v[i]/self.env.weight_volume_ratio <= 0 , name=f"volume_{i}_cons" )

        return model

    def _get_expected_demand(self):

        # get probability of requests over items and periods
        probs = {}
        for i in range(self.n_items):

            idx = i % self.n_classes
            probs[i] = {}
            prob_i = self.env.request_probs[idx]

            for t in range(self.num_periods):
                if t < 20:
                    p_request = 0.90
                    price_probs = [0.7, 0.2, 0.0]
                elif t < 40:
                    p_request = 1.00
                    price_probs = [0.4, 0.2, 0.4]
                else:
                    p_request = 0.90
                    price_probs = [0.0, 0.2, 0.7]

                if i < self.n_classes:
                    price_prob = price_probs[0]
                elif i < 2 * self.n_classes:
                    price_prob = price_probs[1]
                else:
                    price_prob = price_probs[2]

                probs[i][t] = prob_i * price_prob

        # get expected demand
        d = np.zeros(self.n_items)

        for i in range(self.n_items):
            d[i] = sum(probs[i].values())

        return d

    def _get_expected_price(self):
        f = [0.7] * self.n_classes
        f += [1.0] * self.n_classes
        f += [1.4] * self.n_classes

        f = np.array(f)

        return f


    def _get_expected_chargeable_weight(self):
        w = np.zeros(self.n_items)
        for i in range(self.n_items):
            idx = i % self.n_classes
            w[i] = max([self.env.weights[idx], self.env.volumes[idx]/self.env.weight_volume_ratio])

        return w




#--------------------------------------------------------#
#         Dynamic Programming Decomposition (DPD)        #
#--------------------------------------------------------#

class DPD(object):
	
	def __init__(self, env):
		self.env = env
		self.dlp = DLP(env)
		self.dlp.init()
		
		self.n_items = self.dlp.n_items
		self.n_classes =self.dlp.n_classes
		self.n_price_ratios =self.dlp.n_price_ratios
		self.num_periods = self.env.num_periods
		
		self.b_w = self.dlp.b_w
		self.b_v = self.dlp.b_v
		
	def init(self):
		self.init_parameters()
		self.compute_lookup_tables()
		return 
	
	def get_action(self, obs):
		
		if len(obs.action_set) == 1:
			return 0
			
		# set action to 1 if oppurtunity cost is exceeded
		request_price_dict = {0.7 : 0, 1.0 : 1, 1.4 : 2} 
		item = obs.request[0] + self.n_classes * request_price_dict[obs.request[1]]
		idx = item % self.n_classes
		price = self.dlp.f[item] * self.dlp.w[item]
		assert(price == obs.request[1] * self.dlp.w[item])
		
		weight_sum, volume_sum = self.get_utilized_weight_volume()
		weight_sum = int(weight_sum)
		volume_sum = int(volume_sum)
		weight_request = self.env.weights[idx]
		volume_request = self.env.volumes[idx]
		
		t = obs.period
		cost = self.H_w[t+1, weight_sum] - self.H_w[t+1, weight_sum + weight_request] 
		cost += self.H_v[t+1, volume_sum] - self.H_v[t+1, volume_sum + volume_request] 
		
		action = 0
		if price >= cost:
			action = 1
		
		return action
		
	
	def init_parameters(self):
		
		# get split prices
		p = self.dlp.f * self.dlp.w

		p_w = np.zeros(self.n_items)
		for i in range(self.n_items):
			idx = i % self.dlp.n_classes
			p_w[i] = p[i] * (self.b_w * self.env.weights[idx])/(self.b_w * self.env.weights[idx] + self.b_v * self.env.volumes[idx])
		
		p_v = np.zeros(self.n_items)
		for i in range(self.n_items):
			idx = i % self.dlp.n_classes
			p_v[i] = p[i] * (self.b_v * self.env.volumes[idx])/(self.b_w * self.env.weights[idx] + self.b_v * self.env.volumes[idx])
			
		assert(np.abs(np.sum(p_w + p_v - p)) <= 1e-5)
		
		# get split costs
		pi_w = (self.env.cost_multiplier * self.b_w) / (self.b_w + self.b_v)
		pi_v = (self.env.cost_multiplier * self.b_v) / (self.b_w + self.b_v)

		assert(pi_w + pi_v <= self.env.cost_multiplier + 1e-5)
		
		# request probabilties for easy access
		# get probability of requests over items and periods
		probs = {}
		for i in range(self.n_items):
			
			idx = i % self.n_classes
			probs[i] = {}
			prob_i = self.env.request_probs[idx]

			for t in range(self.num_periods):
				if t < 20:
					p_request = 0.90
					price_probs = [0.7, 0.2, 0.0]
				elif t < 40:
					p_request = 1.00
					price_probs = [0.4, 0.2, 0.4]
				else:
					p_request = 0.90
					price_probs = [0.0, 0.2, 0.7]
				
				if i < self.n_classes:
					price_prob = price_probs[0]
				elif i < 2 * self.n_classes:
					price_prob = price_probs[1]
				else:
					price_prob = price_probs[2]
					
				probs[i][t] = prob_i * price_prob
				
		probs[-1] = {}
		for t in range(self.num_periods):
			if t < 20:
				probs[-1][t] = 0.10
			elif t < 40:
				probs[-1][t] = 0.0
			else:
				probs[-1][t] = 0.10     
				
		# check that all probabilities are valid
		for t in range(self.num_periods):
			items = list(range(-1, self.n_items))
			p = list(map(lambda x: probs[x][t], items))
			assert(np.abs(np.sum(p)) - 1 <= 1e-5)
			
		# save parameters
		self.p_w = p_w
		self.p_v = p_v
		self.pi_w = pi_w
		self.pi_v = pi_v
		
		self.probs = probs
		
		return
	
	def compute_lookup_tables(self):
		self.compute_H_w()
		self.compute_H_v()
		
	def compute_H_w(self):
		T = self.num_periods
		W = 3 * int(self.env.expected_weight_capacity)

		H = np.zeros((T+1, W))

		# costs in last period
		for i in range(W):
			H[T, i] = - self.pi_w * max([0, i - self.env.expected_weight_capacity])

		for t in range(T-1, -1, -1):
			for w in range(W):
				sum_ = 0
				sum_ += self.probs[-1][t] * H[t+1, w] # no request

				# iterate over all possible requests
				sum_accept = 0
				sum_reject = 0
				for i in range(self.n_items):
					idx = i % self.n_classes
					if w + self.env.weights[idx] >= W:
						H_t_1_accept = - self.pi_w * (w + self.env.weights[idx] - self.env.expected_weight_capacity)
					else:
						H_t_1_accept = H[t+1, w + self.env.weights[idx]]
					if w >= W:
						H_t_1_reject = - self.pi_w * (w - self.env.expected_weight_capacity)
					else:
						H_t_1_reject = H[t+1, w]                                  
						
					sum_accept += self.probs[i][t] * (self.p_w[i] + H_t_1_accept)
					sum_reject += self.probs[i][t] * (H_t_1_reject)
					
					accept_reward = self.probs[i][t] * (self.p_w[i] + H_t_1_accept)
					reject_reward = self.probs[i][t] * (H_t_1_reject)
					
					#sum_ += max([accept_reward, reject_reward])
					
				sum_ += max([sum_accept, sum_reject])

				H[t, w] = sum_
		
		self.H_w = H
		return

	def compute_H_v(self):
		T = self.num_periods
		V = 3 * int(self.env.expected_volume_capacity)

		H = np.zeros((T+1, V))

		# costs in last period
		for i in range(V):
			H[T, i] = - self.pi_v * max([0, i - self.env.expected_volume_capacity])

		for t in range(T-1, -1, -1):
			for v in range(V):
				sum_ = 0
				sum_ += self.probs[-1][t] * H[t+1, v] # no request

				# iterate over all possible requests
				sum_accept = 0
				sum_reject = 0
				for i in range(self.n_items):
					idx = i % self.n_classes
					if v + self.env.volumes[idx] >= V:
						H_t_1_accept = - self.pi_v * (v + self.env.volumes[idx] - self.env.expected_volume_capacity)
					else:
						H_t_1_accept = H[t+1, v + self.env.volumes[idx]]
					if v >= V:
						H_t_1_reject = - self.pi_v * (v - self.env.expected_volume_capacity)
					else:
						H_t_1_reject = H[t+1, v]                                  
						
					sum_accept += self.probs[i][t] * (self.p_v[i] + H_t_1_accept)
					sum_reject += self.probs[i][t] * (H_t_1_reject)
					
					accept_reward = self.probs[i][t] * (self.p_v[i] + H_t_1_accept)
					reject_reward = self.probs[i][t] * (H_t_1_reject)
					
					#sum_ += max([accept_reward, reject_reward])

				sum_ += max([sum_accept, sum_reject])

				H[t, v] = sum_
		
		self.H_v = H
		return
	
	
	def get_utilized_weight_volume(self):

		# get counts for each item
		item_count = np.zeros(self.env.n_items)
		for key, value in self.env.state.items():
			item_count[key[0]] += value

		used_weight = np.array(self.env.weights) @ np.array(item_count)
		used_volume = np.array(self.env.volumes) @ np.array(item_count)

		return used_weight, used_volume
	
	



		
