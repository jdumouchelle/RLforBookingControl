import numpy as np

from typing import Dict, Tuple, List

class SpotMarketRequests(object):
    
    def __init__(self, 
                 theta_s:float,
                 num_cargo_types:int=1000, 
                 gamma:float = 6.0,
                 shipping_rate_limits_s:Tuple[int] = (1500, 3000),
                 truncation_limits_weight:Tuple[float] = (0.05, 3.5),
                 beta_weight:float = 0.25,
                 truncation_limits_density:Tuple[float] = (0.03, 0.4),
                 mu_density:float = 0.167,
                 sigma_density:float = 0.04,
                ):
        """
        Construct for SpotMarketRequests

        Params
        -------------------------
        """
        
        self.mean_weights = [5, 50, 50, 50, 100, 100, 100, 100, 200, 200, 200, 250,
                            250, 300, 400, 500, 1000, 1500, 2500, 3500, 70, 70, 210, 210]
        self.mean_volumes = [30, 29, 27, 25, 59, 58, 55, 52, 125, 119, 100, 147, 138, 
                            179, 235, 277, 598, 898, 1488, 2083, 233, 17, 700, 52]
        
    def seed(self, seed:int):
        """
        Seeds the spot market requests for reproducibility.  
        
        Params
        -------------------------
        seed:
            The seed whic controls randomization.  
        """
        self.rng.seed(seed)
        
    

    def intialize_distributions(self):
        """
        Initializes all distributions for the cargo and expected distributions
        """
        self.initialize_cargo_distributions()
        self.initialize_expected()
        
    def initialize_cargo_distributions(self):
        """
        Initialize shipping rate, weight, and density distributions.  
        """
        # sample shipping rate for each cargo type
        shipping_rates = self.rng.randint(low=self.shipping_rate_limits_s[0], 
                                          high=self.shipping_rate_limits_s[1], 
                                          size=self.num_cargo_types)
        
        # sample weight of each cargo type
        weights = self.rng.exponential(scale = self.beta_weight, 
                                       size = self.num_cargo_types)

        weights[weights < self.truncation_limits_weight[0]] = self.truncation_limits_weight[0]
        weights[weights > self.truncation_limits_weight[1]] = self.truncation_limits_weight[1]
        
        
        # sample densities of each cargo type
        densities = self.rng.normal(loc = self.mu_density, 
                                    scale = self.sigma_density, 
                                    size = self.num_cargo_types)

        densities[densities < self.truncation_limits_density[0]] = self.truncation_limits_density[0]
        densities[densities > self.truncation_limits_density[1]] = self.truncation_limits_density[1]
        
        self.shipping_rates = shipping_rates
        self.weights = weights
        self.densities = densities
        
        return
        
        
    def initialize_expected(self):
        """
        Initialize expected volumes, revenues, and costs. 
        """
        self.E_volumes = self.weights / self.densities
        
        self.E_revenue = np.zeros(self.num_cargo_types)
        for i in range(self.num_cargo_types):
            self.E_revenue[i] = np.max([self.weights[i], self.E_volumes[i] / self.gamma])
        self.E_revenue *= self.shipping_rates
        
        self.E_cost_shipping = self.shipping_rates * (0.04 * self.E_volumes + 0.4 * self.weights)
        
        self.E_cost_outsourcing = self.shipping_rates * (0.15 * self.E_volumes + 1.5 * self.weights)    
        
        
    def sample(self, cargo_type:int):
        """
        Samples the volume, revenue, and costs of the given cargo type.  
        
        Params
        -------------------------
        cargo_type:
            The cargo type to sample.  In the range [0, num_cargo_types].
        """
        weight = self.weights[cargo_type]
        density = self.densities[cargo_type]
        shipping_rate = self.shipping_rates[cargo_type]
        
        volume = self.rng.lognormal(mean = weight / density, 
                                    sigma = self.theta, 
                                    size = None)
        
        revenue = shipping_rate * np.max([weight, volume / self.gamma])
        
        cost_shipping = shipping_rate * (0.04 * volume + 0.4 * weight)
        cost_outsourcing = shipping_rate * (0.15 * volume + 1.5 * weight)
        
        #cost_outsourcing *= 1.25

        if cost_outsourcing < revenue:
            print('Greater outsourcing cost')
            print('  O:', cost_outsourcing)
            print('  R:', revenue)
            cost_outsourcing = revenue * 1.1


        if cost_shipping > revenue:
            print('Smaller shiping cost')
            print('  S:', cost_outsourcing)
            print('  R:', revenue)
            cost_shipping = revenue * 0.9

        sample_dict = {
            'cargo_tyoe' : cargo_type,
            'shipping_rate' : shipping_rate,
            'weight' : weight,
            'volume' : volume,
            'density' : density,
            'revenue' : revenue,
            'cost_shipping' : cost_shipping,
            'cost_outsourcing' : cost_outsourcing,
        }
        
        return sample_dict