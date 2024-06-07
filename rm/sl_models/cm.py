import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

from rm.envs.cm.utils import get_set_data, get_linear_data


class LinearModel(object):
    
    def __init__(self,
        model, 
        scaler=None):
        """
        Constructor for Linear model

        Params
        -------------------------
        model:
            An estimator to be used in prediction.
        scaler:
            An scaler to be used to transform the input data.
        
        """
        self.model = model
        self.scaler = scaler


    def fit(self, data):
        """
        Fits the model on the data.  

        Params
        -------------------------
        data:
            A list of data.  
        """

        x, y = get_linear_data(data)

        if self.scaler is not None:
             x = self.scaler.fit_transform(x)

        self.model.fit(x, y)


    def predict(self, data):
        """
        Predicts on the data. 

        Params
        -------------------------
        data:
            A list of data points.  
            This should be obtained from MLDataGenerator.  
        """
        if isinstance(data, list):
            x, _ = get_linear_data(data)
        elif isinstance(data, dict):
            x = data['linear_features']
        else:
            raise Exception("Dataset must be one of 'list' or 'dict'.")

        if self.scaler is not None:
            x = self.scaler.transform(x)
            
        preds = self.model.predict(x)

        return preds

    def evaluate(self, data, metric='mse'):
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
        _, y = get_linear_data(data)
        preds = self.predict(data)

        if metric == 'mse':
            score = MSE(y, preds)
        elif metric  == 'mae':
            score = MAE(y, preds)    
        else:
            raise Exception('metric not yet implemented')
            
        return score



class SetNetwork(nn.Module):
    """
        Deepset based input invariant model neural network.  
    """
    def __init__(self, 
                 inst_input_dim,
                 req_input_dim, 
                 req_hidden_dim, 
                 req_embed_dim1, 
                 req_embed_dim2, 
                 concat_hidden_dim,
                 activation_fn,
                 dropout=0):
        """
            Builds a neural network from a list of hidden dimensions.  
            If the list is empty, then the model is simply linear regression. 
        """
        super(SetNetwork, self).__init__()

        self.inst_input_dim = inst_input_dim

        self.req_input_dim = req_input_dim
        self.req_hidden_dim = req_hidden_dim
        self.req_embed_dim1 = req_embed_dim1
        self.req_embed_dim2 = req_embed_dim2

        self.concat_hidden_dim = concat_hidden_dim

        self.output_dim = 1
        
        self.activation_fn = activation_fn

        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(p=dropout)

        # layers for scenario input 
        #   no bias, so that these layers can be better parallelized, for instances with varing 
        #   numbers of requests.
        self.req_input = nn.Linear(self.req_input_dim, self.req_hidden_dim, bias=False)
        self.req_embed1 = nn.Linear(self.req_hidden_dim, self.req_embed_dim1, bias=False)
        self.req_embed2 = nn.Linear(self.req_embed_dim1, self.req_embed_dim2, bias=False)

        # for concatenation layer
        self.concat_input = nn.Linear(self.inst_input_dim + self.req_embed_dim2, self.concat_hidden_dim)
        self.concat_output = nn.Linear(self.concat_hidden_dim, self.output_dim)


    def forward(self, x_inst, x_req, x_n_req=None):
        """ Forward pass. """

        # embed scenarios
        x_req_embed = self.embed_requests(x_req, x_n_req)

        # concat first stage solution and scenario embedding
        x = torch.cat((x_inst, x_req_embed), 1)

        # get aggregate prediction
        x = self.concat_input(x)
        x = self.activation_fn(x)
        if self.use_dropout:
            x = self.dropout_layer(x)
        x = self.concat_output(x)

        return x

    def embed_requests(self, x_req, x_n_req=None):
        """ Scenario embedding. """
        # for each batch, pass-non padded values in. 
        if x_n_req is not None:

            x_req_embed = []
            for i in range(x_req.shape[0]):
                n_req = int(x_n_req[i].item())
                x_req_in = x_req[i,:n_req]
                x_req_in = torch.reshape(x_req_in, (1, x_req_in.shape[0], x_req_in.shape[1]))

                # embed 
                x_req_embed_i = self.req_input(x_req_in)
                x_req_embed_i = self.activation_fn(x_req_embed_i)
                if self.use_dropout:
                    x_req_embed_i = self.dropout_layer(x_req_embed_i)

                x_req_embed_i = self.req_embed1(x_req_embed_i)
                x_req_embed_i = self.activation_fn(x_req_embed_i)
                if self.use_dropout:
                    x_req_embed_i = self.dropout_layer(x_req_embed_i)

                x_req_embed_i = torch.sum(x_req_embed_i, axis=1) # concat all inputs

                x_req_embed_i = self.req_embed2(x_req_embed_i)
                x_req_embed_i = self.activation_fn(x_req_embed_i)
                if self.use_dropout:
                    x_req_embed_i = self.dropout_layer(x_req_embed_i)

                x_req_embed.append(x_req_embed_i)

            x_req_embed = torch.stack(x_req_embed)
            x_req_embed = torch.reshape(x_req_embed, (x_req_embed.shape[0], x_req_embed.shape[2]))

        # assume no padding, i.e. full scenario set
        else:
            
            x_req_embed = self.req_input(x_req)
            x_req_embed = self.activation_fn(x_req_embed)
            if self.use_dropout:
                x_req_embed = self.dropout_layer(x_req_embed)

            x_req_embed = self.req_embed1(x_req_embed)
            x_req_embed =  self.activation_fn(x_req_embed)
            if self.use_dropout:
                x_req_embed = self.dropout_layer(x_req_embed)

            x_req_embed = torch.sum(x_req_embed, axis=1) # concat all inputs

            x_req_embed = self.req_embed2(x_req_embed)
            x_req_embed = self.activation_fn(x_req_embed)
            if self.use_dropout:
                x_req_embed = self.dropout_layer(x_req_embed)

        return x_req_embed

    def predict(self, dataset, max_requests=60):
        """ Prediccts operational cost for a dataset.  """
        if isinstance(dataset, list):
            x_inst, x_req, _, _ = get_set_data(dataset, max_requests)

        elif isinstance(dataset, dict):
            #print(dataset)
            x_inst = dataset['set_features']['x_inst']
            x_req = dataset['set_features']['x_req']

        else:
            raise Exception("Dataset must be one of 'list' or 'dict'.")
        
        if torch.cuda.is_available():
            x_inst, x_req = x_inst.cuda(), x_req.cuda()
            
        preds = self.forward(x_inst, x_req)
        preds = preds.cpu().detach().numpy()
        
        return preds