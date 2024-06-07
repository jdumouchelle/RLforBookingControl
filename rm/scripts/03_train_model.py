import copy
import time
import pickle
import argparse
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE


import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from rm import params
from rm.utils import factory_get_path, factory_env, factory_linear_model, factory_input_invariant_model


#--------------------------------------------------------#
#                  Instance/data loading                 #
#--------------------------------------------------------#

def load_inst(cfg, args):
    """ Loads instances.  """
    inst_fp = get_path(cfg, 'inst', args.data_dir)
    with open(inst_fp, 'rb') as p:
        env = pickle.load(p)
    return env


def load_data(cfg, args):
    """ Loads machine learning data.  """ 
    fp_data = get_path(cfg, 'ml_data', args.data_dir)
    with open(fp_data, 'rb') as p:
        data = pickle.load(p)
    return data 



#--------------------------------------------------------#
#                  Model initialization                  #
#--------------------------------------------------------#

def init_lr(data, env, cfg, args):
    cv = 10
    lr_grid = {"l1_ratio": [.1, .5, .7, .9, .95, .99, 1]}
    lr = ElasticNetCV(l1_ratio = lr_grid["l1_ratio"], cv=5)
    return lr


def init_rf(data, env, cfg, args):
    """ Initializes random search RN model.  """
    # init RF random search model
    rf_grid = {"n_estimators": np.arange(10, 100, 10),
               "max_depth": [None, 3, 5, 10],
               "min_samples_split": np.arange(2, 20, 2),
               "min_samples_leaf": np.arange(1, 20, 2)}

    # init RF random search model
    rf = RandomizedSearchCV(RFR(n_jobs=-1, random_state=cfg.seed),
                                  param_distributions=rf_grid,
                                  n_iter=50,
                                  cv=10,
                                  verbose=True)

    return rf


#--------------------------------------------------------#
#                      Model Training                    #
#--------------------------------------------------------#

def train_linear_model(data, env, cfg, args):
    """ Trains a linear model.  """
    if args.model_type == "lr":
        base_model = init_lr(data, env, cfg, args)
    elif args.model_type == "rf":
        base_model = init_rf(data, env, cfg, args)
    else:
        raise Exception(f"Not a valid model_type: {args.model_type}")

    tr_data = data['tr_data']
    val_data = data['val_data']

    # get model based on problem class
    if "vrp" in args.problem:
        ml_model = LinearModel(
            env = env, 
            model = base_model, 
            scaler = MinMaxScaler(), 
            adjustment_method = 'bpp')

    elif "cm" in args.problem:
        ml_model = LinearModel(base_model, MinMaxScaler())

    # fit
    results = {}

    train_time = time.time()
    ml_model.fit(tr_data)
    train_time = time.time() - train_time

    # evaluate
    results['tr_mae'] = ml_model.evaluate(tr_data, metric='mae')
    results['tr_mse'] = ml_model.evaluate(tr_data, metric='mse')
    results['val_mae'] = ml_model.evaluate(val_data, metric='mae')
    results['val_mse'] = ml_model.evaluate(val_data, metric='mse')
    results['time'] = train_time

    print(f"  Results for {args.model_type}: ")
    print("    Train MAE:      ", results['tr_mae'])
    print("    Train MSE:      ", results['tr_mse'])
    print("    Validation MAE: ", results['val_mae'])
    print("    Validation MSE: ", results['val_mse'])

    return ml_model, results


def train_graph_network(data, env, cfg, args):
    """ Trains graph based nerual network. """
    from rm.envs.vrp.utils import get_graph_data
    torch.manual_seed(cfg.seed)

    # optimization
    criteria = nn.MSELoss() # fixed
    opt = optim.Adam        # fixed
    if args.activation_fn == "sigmoid" or args.activation_fn == "tanh":
        activation_fn = getattr(torch, args.activation_fn)
    else:
        activation_fn = getattr(F, args.activation_fn)

    # dimensions
    inst_input_dim = 5 # fixed
    iv_input_dim = 3 # fixed

    model = IIVModel(inst_input_dim, 
                        iv_input_dim, 
                        args.iv_hidden_dim, 
                        args.iv_embed_dim1, 
                        args.iv_embed_dim2, 
                        args.concat_hidden_dim,
                        activation_fn, 
                        env,
                        dropout=args.dropout)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = opt(model.parameters(), lr=args.lr)

    # get tensor dataset
    tr_data = data['tr_data']
    val_data = data['val_data']

    x_inst_tr, x_iv_tr, x_n_iv_tr, y_tr = get_graph_data(tr_data)
    x_inst_val, x_iv_val, x_n_iv_val, y_val = get_graph_data(val_data)

    if torch.cuda.is_available():
        x_inst_tr, x_iv_tr, x_n_iv_tr, y_tr = x_inst_tr.cuda(), x_iv_tr.cuda(), x_n_iv_tr.cuda(), y_tr.cuda() 
        x_inst_val, x_iv_val, x_n_iv_val, y_val = x_inst_val.cuda(), x_iv_val.cuda(), x_n_iv_val.cuda(), y_val.cuda() 

    dataset_tr = TensorDataset(x_inst_tr, x_iv_tr, x_n_iv_tr, y_tr)
    dataset_val = TensorDataset(x_inst_val, x_iv_val, x_n_iv_val, y_val)
    loader_tr = DataLoader(dataset_tr, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    # get true adjusted labels
    label_report = 'adjusted_operational_cost'
    y_tr_report = np.array(list(map(lambda x: x[label_report], tr_data)))
    y_val_report = np.array(list(map(lambda x: x[label_report], val_data)))

    label_eval = 'operational_cost'
    y_tr_eval = np.array(list(map(lambda x: x[label_eval], tr_data)))
    y_val_eval = np.array(list(map(lambda x: x[label_eval], val_data)))

    # train
    best_model = None
    best_mae = 1e7

    tr_results = {'loss' : [], 'mse' : [], 'mae' : []}
    val_results = {'mse' : [], 'mae' : []}

    nn_time = time.time()

    for epoch in range(args.n_epochs):
        model.train()
        ep_loss = []
        for i, data in enumerate(loader_tr):

            x_inst, x_iv, x_n_iv, y = data
                
            preds = model(x_inst, x_iv, None)            
            loss = criteria(preds, y)

            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()
            
            ep_loss.append(loss.item())
        
        tr_results['loss'].append(np.mean(ep_loss))
            
        # get validation results
        if ((epoch + 1) % args.eval_freq) == 0:
            model.eval()
            tr_preds = model.predict(tr_data, predict_adjusted=False)
            tr_mse = MSE(y_tr_eval, tr_preds)
            tr_mae = MAE(y_tr_eval, tr_preds)

            tr_results['mse'].append(tr_mse)
            tr_results['mae'].append(tr_mae)
            
            val_preds = model.predict(val_data, predict_adjusted=False)
            val_mse = MSE(y_val_eval, val_preds)
            val_mae = MAE(y_val_eval, val_preds)

            val_results['mse'].append(val_mse)
            val_results['mae'].append(val_mae)
            
            print('[%d] MSE:   tr: %.3f, val: %.3f' %
                  (epoch + 1, tr_mse, val_mse))
            print('     MAE:   tr: %.3f, val: %.3f' %
                  (tr_mae, val_mae))

            if best_mae > val_mae:
                print('     New best model')
                best_mae = val_mae
                best_model = copy.deepcopy(model)
                
                best_tr_mae = tr_mae
                best_tr_mse = tr_mse 
                best_val_mae = val_mae
                best_val_mse = val_mse

    nn_time = time.time() - nn_time

    print('Finished Training')
    print("NN time:", nn_time)
    
    nn_results = {
        'tr_mae' : best_tr_mae,
        'tr_mse' : best_tr_mse,
        'val_mae' : best_val_mae,
        'val_mse' : best_val_mse,
        'time' : nn_time,
        'tr_results' : tr_results,
        'val_results' : val_results,
        'params' : vars(args)
    }

    # model to evaluate mode
    best_model.eval()

    return best_model, nn_results


def train_neural_model(data, cfg, args):
    """ Trains set-based nerual network. """
    print("Training neural model... ")
    from rm.envs.cm.utils import get_set_data

    torch.manual_seed(cfg.seed)

    if cfg.single_leg:
        inst_input_dim = 3  # fixed for all single leg problems
        iv_input_dim = 3   # fixed for all single leg problems
    else:
        raise Exception("Multiple Leg feature dimension not yet defined!")


    tr_data = data['tr_data']
    val_data = data['val_data']

    # fixed hyperparameters (i.e. not changeable in argparse)
    criteria = nn.L1Loss() 
    opt = optim.Adam
    if args.activation_fn == "sigmoid" or args.activation_fn == "tanh":
        activation_fn = getattr(torch, args.activation_fn)
    else:
        activation_fn = getattr(F, args.activation_fn)

    # initialize model
    model = IIVModel(
        inst_input_dim,
        iv_input_dim, 
        args.iv_hidden_dim, 
        args.iv_embed_dim1, 
        args.iv_embed_dim2, 
        args.concat_hidden_dim,
        activation_fn,
        dropout=args.dropout)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = opt(model.parameters(), lr=args.lr)

    # get data 
    x_inst_tr, x_iv_tr, x_n_iv_tr, y_tr = get_set_data(tr_data)
    x_inst_val, x_iv_val, x_n_iv_val, y_val = get_set_data(val_data)

    if torch.cuda.is_available():
        x_inst_tr, x_iv_tr, x_n_iv_tr, y_tr = x_inst_tr.cuda(), x_iv_tr.cuda(), x_n_iv_tr.cuda(), y_tr.cuda() 
        x_inst_val, x_iv_val, x_n_iv_val, y_val = x_inst_val.cuda(), x_iv_val.cuda(), x_n_iv_val.cuda(), y_val.cuda() 

    dataset_tr = TensorDataset(x_inst_tr, x_iv_tr, x_n_iv_tr, y_tr)
    dataset_val = TensorDataset(x_inst_val, x_iv_val, x_n_iv_val, y_val)

    loader_tr = DataLoader(dataset_tr, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    y_tr_np = y_tr.cpu().numpy()
    y_val_np = y_val.cpu().numpy()

    # train model
    best_model = None
    best_mae = 1e7

    tr_results = {'loss' : [], 'mse' : [], 'mae' : []}
    val_results = {'mse' : [], 'mae' : []}

    nn_time = time.time()

    for epoch in range(args.n_epochs):

        model.train()
        ep_loss = []
        for i, data in enumerate(loader_tr):

            x_inst, x_iv, x_n_iv, y = data
                
            preds = model(x_inst, x_iv, None)
            
            loss = criteria(preds, y)

            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()
            
            ep_loss.append(loss.item())
        
        tr_results['loss'].append(np.mean(ep_loss))
            
        # get validation results
        if ((epoch + 1) % args.eval_freq) == 0:
            
            model.eval()
            # train preds
            tr_preds = model.predict(tr_data)
            tr_mse = MSE(y_tr_np, tr_preds)
            tr_mae = MAE(y_tr_np, tr_preds)
            tr_results['mse'].append(tr_mse)
            tr_results['mae'].append(tr_mae)
            
            # validation preds
            val_preds = model.predict(val_data)      
            val_mse = MSE(y_val_np, val_preds)
            val_mae = MAE(y_val_np, val_preds)
            val_results['mse'].append(val_mse)
            val_results['mae'].append(val_mae)
            
            print('  [%d] MSE:   tr: %.3f, val: %.3f' %
                  (epoch + 1, tr_mse, val_mse))
            print('       MAE:   tr: %.3f, val: %.3f' %
                  (tr_mae, val_mae))

            if best_mae > val_mae:
                print('       New best model')
                best_mae = val_mae
                best_model = copy.deepcopy(model)

                best_tr_mae = tr_mae
                best_tr_mse = tr_mse 
                best_val_mae = val_mae
                best_val_mse = val_mse

    nn_time = time.time() - nn_time

    print('Finished Training')
    print("NN time:", nn_time)

    nn_results = {
        'tr_mae' : best_tr_mae,
        'tr_mse' : best_tr_mse,
        'val_mae' : best_val_mae,
        'val_mse' : best_val_mse,
        'time' : nn_time,
        'tr_results' : tr_results,
        'val_results' : val_results,
        'params' : vars(args)
    }

    # model to evaluate mode
    best_model.eval()

    return best_model, nn_results



#--------------------------------------------------------#
#                Model-specific path                     #
#--------------------------------------------------------#

def get_config_str(args):
    """ Gets unique string base on the args passed.  """
    config_str = f"lr-{args.lr}_"
    config_str += f"act-{args.activation_fn}_"
    config_str += f"do-{args.dropout}_"
    config_str += f"bs-{args.batch_size}_"
    config_str += f"e1hd-{args.iv_hidden_dim}_"
    config_str += f"e2hd-{args.iv_embed_dim1}_"
    config_str += f"e3hd-{args.iv_embed_dim2}_"
    config_str += f"chd-{args.concat_hidden_dim}"

    return config_str



#--------------------------------------------------------#
#                          Main                          #
#--------------------------------------------------------#

def main(args):
    """ Trains model of specified type. """
    global get_path, Environment, LinearModel, IIVModel
    get_path = factory_get_path(args)
    Environment = factory_env(args)
    LinearModel = factory_linear_model(args)
    IIVModel = factory_input_invariant_model(args)

    cfg = getattr(params, args.problem)
    
    # load instance and data
    env = load_inst(cfg, args)
    data = load_data(cfg, args)

    # train model
    if args.model_type == "nn" and "vrp" in args.problem:
        ml_model, results = train_graph_network(data, env, cfg, args)
    
    elif args.model_type == "nn" and "cm" in args.problem:
        ml_model, results = train_neural_model(data, cfg, args)

    else:
        ml_model, results = train_linear_model(data, env, cfg, args)

    # get problem config str (only applicable for NN)
    config_str = get_config_str(args)
    
    # save trained model
    if "nn" == args.model_type:
        fp_model =  get_path(cfg, f'random_search_nn/{args.model_type}', args.data_dir, suffix=f'__{config_str}__.pt')
        fp_results =  get_path(cfg, f'random_search_nn/{args.model_type}_results', args.data_dir, suffix=f'__{config_str}__.pkl')

        torch.save(ml_model, fp_model)

    else:
        fp_model =  get_path(cfg, f'{args.model_type}', args.data_dir)
        fp_results =  get_path(cfg, f'{args.model_type}_results', args.data_dir)
        with open(fp_model, "wb") as p:
            pickle.dump(ml_model, p)

    # save results for model
    with open(fp_results, 'wb') as p:
        pickle.dump(results, p)

    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train supervised learning models.')

    parser.add_argument('--data_dir', type=str, default="./data/", help='Data directory root.')
    parser.add_argument('--problem', type=str, help='Problem class.')
    parser.add_argument('--model_type', type=str, help='Type of learning model.')

    ## Optional NN hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.001, help='Dropout rate.')
    parser.add_argument('--activation_fn', type=str, default="relu", help='Learning rate.')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--eval_freq', type=int, default=10, help='Evaluation frequency.')

    parser.add_argument('--iv_hidden_dim', type=int, default=256, help='Location hidden dim.')
    parser.add_argument('--iv_embed_dim1', type=int, default=64, help='Location embed dim 1.')
    parser.add_argument('--iv_embed_dim2', type=int, default=8, help='Location embed dim 2.')
    parser.add_argument('--concat_hidden_dim', type=int, default=512, help='Concat dim.')

    args = parser.parse_args()

    main(args)