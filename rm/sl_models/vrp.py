import sys
import numpy as np
import multiprocessing as mp

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

from rm.envs.vrp.utils import get_graph_data
from rm.envs.vrp.bpp_solver import BPPSolver
from rm.envs.vrp.environment import Environment

from typing import Dict, List, Tuple


class LinearModel(object):
    
    def __init__(self, 
        env:"Environment", 
        model:"", 
        scaler:""=None, 
        adjustment_method:str=None, 
        adjustment_model:""=None):
        """
        Constructor for LinearModel.  

        Params
        -------------------------
        env:
            The initialized environment.
        model:
            An estimator to be used in prediction.
        scaler:
            An scaler to be used to transform the input data.
        adjustment_method:
            A string indicating how to adjust for outsourcing costs.  Can be one
            of None, 'ml', or 'bpp'
        adjustment_model:
            A model to be used in predicting the number of vehichles.  
            Only needs to be specified if adjustment_method is ml. 
        """
        self.env = env
        self.model = model
        self.scaler = scaler
        self.adjustment_method = adjustment_method
        self.adjustment_model = adjustment_model
        self.bpp_solver = BPPSolver()

        if self.adjustment_method == 'ml':
            assert(self.adjustment_model is not None)
        

    def fit(self, data:List[Dict]):
        """
        Fits the model on the data.  

        Params
        -------------------------
        data:
            A list of data points.  
            This should be obtained from MLDataGenerator.  
        """
        # get labels and features
        if self.adjustment_method == 'bpp' or self.adjustment_method == 'ml':
            label = 'operational_cost'
        else:
            label = 'adjusted_operational_cost'

        X = np.array(list(map(lambda x: list(x['features'].values()), data)))
        y = np.array(list(map(lambda x: x[label], data)))

        if self.scaler is not None:
             X = self.scaler.fit_transform(X)

        self.model.fit(X, y)

        if self.adjustment_method == 'ml':
            y_num_vechicles =  np.array(list(map(lambda x: x['num_vehicles_used'], data)))
            self.adjustment_model.fit(X, y_num_vechicles)

        return


    def predict(self, data:List[Dict]):
        """
        Predicts on the data. 

        Params
        -------------------------
        data:
            A list of data points.  
            This should be obtained from MLDataGenerator.  
        """
        X = np.array(list(map(lambda x: list(x['features'].values()), data)))

        if self.scaler is not None:
            X = self.scaler.transform(X)
            
        preds = self.model.predict(X)

        # solve bpp problem for each instance
        if self.adjustment_method == 'bpp':
            bpp_results = []
            for i in range(len(data)):
                num_vehicles_used = self.bpp_solver.solve(data[i])
                #if num_vehicles_used < data[i]['num_vehicles_used']:
                    #       print("ERR")
                #   print(i, num_vehicles_used, data[i]['num_vehicles_used'], data[i] )

                #   return              
                if num_vehicles_used > self.env.num_vehicles:
                    preds[i] += self.env.added_vehicle_cost * (num_vehicles_used - self.env.num_vehicles)

        elif self.adjustment_method == 'ml':
            preds_adjustment = self.adjustment_model.predict(X)
            for i in range(preds.shape[0]):
                num_vehicles_estimate = preds_adjustment[i]
                if num_vehicles_estimate > self.env.num_vehicles:
                    preds[i] += self.env.added_vehicle_cost * (num_vehicles_estimate - self.env.num_vehicles)

        return preds


    def evaluate(self, data:List[Dict], metric:str='mse'):
        """
        Evaluates the regression performance on the data.   

        Params
        -------------------------
        data:
            A list of data points.  
            This should be obtained from MLDataGenerator.  
        metric:
            the evaluation metric.  One of 'mse', 'mae', 'mape', 'data'
        """
        # get labels and features
        label = 'adjusted_operational_cost'

        y = np.array(list(map(lambda x: x[label], data)))

        preds = self.predict(data)

        if metric == 'mse':
            score = MSE(y, preds)
            
        elif metric  == 'mae':
            score = MAE(y, preds)

        elif metric == 'mape':
            score = self.mean_absolute_percentage_error(y, preds)

        elif metric == 'data':
            return y, preds
            
        else:
            raise Exception('metric not yet implemented')
            
        return score


    def mean_absolute_percentage_error(self, y_true:np.ndarray, y_pred:np.ndarray): 
        """
        Computes the mean absolute percent error.  

        Params
        -------------------------
        y_true:
            An array containing the true values.
        y_pred:
            An array containing the predicted values.
        """

        return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred))) * 100




class GraphNetwork(nn.Module):
    """
        Graph neural network for supervised learning predictions.
    """
    def __init__(self, 
                inst_input_dim, 
                loc_input_dim, 
                loc_hidden_dim, 
                loc_embed_dim1, 
                loc_embed_dim2, 
                concat_hidden_dim,
                activation_fn,
                env, 
                adjustment_method='bpp',
                dropout=0):
        """
            Builds a neural network from a list of hidden dimensions.  
            If the list is empty, then the model is simply linear regression. 
        """
        super(GraphNetwork, self).__init__()

        self.inst_input_dim = inst_input_dim

        self.loc_input_dim = loc_input_dim
        self.loc_hidden_dim = loc_hidden_dim
        self.loc_embed_dim1 = loc_embed_dim1
        self.loc_embed_dim2 = loc_embed_dim2

        self.concat_hidden_dim = concat_hidden_dim

        self.output_dim = 1
        
        self.activation_fn = activation_fn

        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(p=dropout)
        
        # problem specific infomation
        self.env = env
        self.adjustment_method = adjustment_method
        self.bpp_solver = BPPSolver()
        
        # raise exception as this is only implemented for BPP at the moment.
        if self.adjustment_method != 'bpp': 
            raise Exception("GNN only implemeneted for bpp adjustment")
        
        # layers for scenario input
        self.loc_input = nn.Linear(self.loc_input_dim, self.loc_hidden_dim, bias=False)
        self.loc_embed1 = nn.Linear(self.loc_hidden_dim, self.loc_embed_dim1, bias=False)
        self.loc_embed2 = nn.Linear(self.loc_embed_dim1, self.loc_embed_dim2, bias=False)

        # for concatenation layer
        self.concat_input = nn.Linear(self.inst_input_dim + self.loc_embed_dim2, self.concat_hidden_dim)
        self.concat_output = nn.Linear(self.concat_hidden_dim, self.output_dim)


    def forward(self, x_inst, x_loc, x_n_loc=None):
        """ Forward pass. """

        # embed scenarios
        x_loc_embed = self.embed_locations(x_loc, x_n_loc)

        # concat first stage solution and scenario embedding
        x = torch.cat((x_inst, x_loc_embed), 1)

        # get aggregate prediction
        x = self.concat_input(x)
        x = self.activation_fn(x)
        if self.use_dropout:
            x = self.dropout_layer(x)

        x = self.concat_output(x)

        return x


    def embed_locations(self, x_loc, x_n_loc=None):
        """ Scenario embedding. """
        # for each batch, pass-non padded values in. 
        if x_n_loc is not None:

            x_loc_embed = []
            for i in range(x_loc.shape[0]):
                n_loc = int(x_n_loc[i].item())
                x_loc_in = x_loc[i,:n_loc]
                x_loc_in = torch.reshape(x_loc_in, (1, x_loc_in.shape[0], x_loc_in.shape[1]))

                # embed
                x_loc_embed_i = self.loc_input(x_loc_in)
                x_loc_embed_i = self.activation_fn(x_loc_embed_i)
                if self.use_dropout:
                    x_loc_embed_i = self.dropout_layer(x_loc_embed_i)

                x_loc_embed_i = self.loc_embed1(x_loc_embed_i)
                x_loc_embed_i = self.activation_fn(x_loc_embed_i)
                if self.use_dropout:
                    x_loc_embed_i = self.dropout_layer(x_loc_embed_i)

                x_loc_embed_i = torch.sum(x_loc_embed_i, axis=1) # concat all inputs

                x_loc_embed_i = self.loc_embed2(x_loc_embed_i)
                x_loc_embed_i = self.activation_fn(x_loc_embed_i)
                if self.use_dropout:
                    x_loc_embed_i = self.dropout_layer(x_loc_embed_i)

                x_loc_embed.append(x_loc_embed_i)

            x_loc_embed = torch.stack(x_loc_embed)
            x_loc_embed = torch.reshape(x_loc_embed, (x_loc_embed.shape[0], x_loc_embed.shape[2]))

        # assume no padding, i.e. full scenario set
        else:
            
            x_loc_embed = self.loc_input(x_loc)
            x_loc_embed = self.activation_fn(x_loc_embed)
            if self.use_dropout:
                x_loc_embed = self.dropout_layer(x_loc_embed)

            x_loc_embed = self.loc_embed1(x_loc_embed)
            x_loc_embed =  self.activation_fn(x_loc_embed)
            if self.use_dropout:
                x_loc_embed = self.dropout_layer(x_loc_embed)

            x_loc_embed = torch.sum(x_loc_embed, axis=1) # concat all inputs

            x_loc_embed = self.loc_embed2(x_loc_embed)
            x_loc_embed = self.activation_fn(x_loc_embed)
            if self.use_dropout:
                x_loc_embed = self.dropout_layer(x_loc_embed)

        return x_loc_embed
    

    def predict(self, dataset, predict_adjusted=True):
        """ Prediccts actual operational cost w/ BPP solver.  """
        def get_additional_cost(dataset_i):
            n_vehicles_used = self.bpp_solver.solve(dataset_i)
            if n_vehicles_used > self.env.num_vehicles:
                return self.env.added_vehicle_cost * (n_vehicles_used - self.env.num_vehicles)
            else:
                return 0.0
        
        x_inst, x_loc, _, _ = get_graph_data(dataset)
        
        if torch.cuda.is_available():
            x_inst, x_loc = x_inst.cuda(), x_loc.cuda()
            
        preds = self.forward(x_inst, x_loc)
        preds = preds.cpu().detach().numpy()
        
        # get adjusted based on excess vehicles
        if predict_adjusted:
            additional_costs = np.array(list(map(lambda x: get_additional_cost(x), dataset))).reshape(-1, 1)
            #additional_costs = get_additional_cost_mp(dataset, self.bpp_solver, self.env.num_vehicles, self.env.added_vehicle_cost)
            preds += additional_costs

        return preds


def get_additional_cost_mp(dataset, bpp_solver, num_vehicles, added_vehicle_cost):
    """ Multiprocessing for computing bin-packing based costs.  Not used.  """
    # get number of excess vehicles
    pool = mp.Pool()
    results = []
    for x in dataset:
        results.append(pool.apply_async(get_additional_cost_worker, (x, bpp_solver)))
    results = [r.get() for r in results]
    results = np.array(results).reshape(-1, 1)

    # compute additional vehicle cost
    mask = results > num_vehicles
    results[mask] -= num_vehicles
    results[mask] *= added_vehicle_cost
    return results


def get_additional_cost_worker(x, bpp_solver):
    """ Worker for MP.  """
    n_vehicles_used = bpp_solver.solve(x)
    return n_vehicles_used