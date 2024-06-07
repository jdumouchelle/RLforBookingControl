import sys
import time
import numpy as np

from typing import Dict, List, Tuple


class BPPSolver(object):
    
    def __init__(self, solver_type="gurobi"):
        """
        Constructor for BPP solver.  Takes in the tyoe of solver,
        either mtp or gurobi.  
        """
        self.solver_type = solver_type
    
    
    def solve(self, data:Dict):
        """
        Solves an instance assosiated with the data passed.

        
        Params
        -------------------------
        data:
            the data which the bpp solver uses.  The parameter must be a
            dict containing 'capacity' and 'weights_bpp' as keys.  
        """
        
        # formulate obs demands as a bpp problem
        capacity = int(data['capacity'])
        weights = sorted(data['weights_bpp'], reverse=True)

        num_items = len(weights)
        
        if self.solver_type == "gurobi":
            sol = self._solve_gp(num_items, weights, capacity)
            sol = int(sol)

        return sol
            

    def _solve_gp(self, num_items:int, weights:List, capacity:int):
        """
        Internal function which actually calls MTP.  

        Params
        -------------------------
        num_items:
            the number of items
        weights:
            the weights 
        capacity:
            the capacity 

        """
        import gurobipy as gp

        n_bins_max = int(2 * np.ceil(np.sum(weights) / capacity))

        model = gp.Model()
        model.setParam("OutputFlag", 0)

        x, y = {}, {}

        # variable for each bin
        for j in range(n_bins_max):
            y[j] = model.addVar(name=f"y_{j}", vtype="B", obj=1)

        # variable for the assignment of item i to bin j
        for i in range(num_items):
            x[i] = {}
            for j in range(n_bins_max):
                x[i][j] = model.addVar(name=f"x_{i}_{j}", vtype="B", obj=0)

        # pack only if open
        for i in range(num_items):
            for j in range(n_bins_max):
                model.addConstr(x[i][j] <= y[j], f"bound_{i}_{j}")

        # capacity constraints
        for j in range(n_bins_max):
            eq_ = 0
            for i in range(num_items):
                eq_ += weights[i] * x[i][j]
            model.addConstr(eq_ <= y[j] * capacity, f"cap_{j}")

        # is packed constraints
        for i in range(num_items):
            eq_ = 0
            for j in range(n_bins_max):
                eq_ += x[i][j]
            model.addConstr(eq_ == 1, f"packed_{i}")
        
        model.optimize()
       
        return model.objVal
