import os
import copy
import time
import pickle
import numpy as np

import gym
from gym import spaces

import torch
from torch import nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict


class SetNetwork(TorchModelV2, nn.Module):
    """ Generic fully connected network. """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # check to make sure correct params were set  in config
        assert(model_config["custom_model_config"]["concat_hidden_dim"] == num_outputs)

        # store original observation space
        self.orig_space = obs_space.original_space
        
        # input dimensions
        self.inst_input_dim = self.orig_space["inst_obs"].shape[0]
        self.iv_input_dim = self.orig_space["req_obs"].shape[1]
        
        # network architechure
        self.iv_hidden_dim = model_config["custom_model_config"]["iv_hidden_dim"]
        self.iv_embed_dim1 = model_config["custom_model_config"]["iv_embed_dim1"]
        self.iv_embed_dim2 = model_config["custom_model_config"]["iv_embed_dim2"]
        self.concat_hidden_dim = model_config["custom_model_config"]["concat_hidden_dim"]
                    
        # additional network params
        activation = model_config["custom_model_config"]["activation_fn"]
        
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")
        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2

        layers = []
        self._logits = None

        # Input Invariant layers
        self.iv_input = SlimFC(in_size=self.iv_input_dim,
                                out_size=self.iv_hidden_dim,
                                initializer=normc_initializer(1.0),
                                activation_fn=activation,
                                use_bias=False)
        
        self.iv_embed1 = SlimFC(in_size=self.iv_hidden_dim,
                                out_size=self.iv_embed_dim1,
                                initializer=normc_initializer(1.0),
                                activation_fn=activation,
                                use_bias=False)
        
        self.iv_embed2 = SlimFC(in_size=self.iv_embed_dim1,
                                out_size=self.iv_embed_dim2,
                                initializer=normc_initializer(1.0),
                                activation_fn=activation,
                                use_bias=False)
        
        # Concatenation Layers
        self.concat_layer = SlimFC(in_size = self.inst_input_dim + self.iv_embed_dim2, 
                                   out_size=self.concat_hidden_dim,
                                   initializer=normc_initializer(1.0),
                                   activation_fn=activation)
        
        prev_layer_size = num_outputs

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)
                    
        # Value function
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            # Input Invariant layers
            self.vf_iv_input = SlimFC(in_size=self.iv_input_dim,
                                    out_size=self.iv_hidden_dim,
                                    initializer=normc_initializer(1.0),
                                    activation_fn=activation,
                                    use_bias=False)

            self.vf_iv_embed1 = SlimFC(in_size=self.iv_hidden_dim,
                                    out_size=self.iv_embed_dim1,
                                    initializer=normc_initializer(1.0),
                                    activation_fn=activation,
                                    use_bias=False)

            self.vf_iv_embed2 = SlimFC(in_size=self.iv_embed_dim1,
                                    out_size=self.iv_embed_dim2,
                                    initializer=normc_initializer(1.0),
                                    activation_fn=activation,
                                    use_bias=False)

            # Concatenation Layers
            self.vf_concat_layer = SlimFC(in_size = self.inst_input_dim + self.iv_embed_dim2, 
                                       out_size=self.concat_hidden_dim,
                                       initializer=normc_initializer(1.0),
                                       activation_fn=activation)
                    
        # vf output layer
        self.vf_output = SlimFC(
                    in_size=num_outputs,
                    out_size=self.concat_hidden_dim,
                    activation_fn=None,
                    initializer=normc_initializer(1.0),
        )
            
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_in = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """ Forward Pass. """        
        self._last_in = input_dict["obs"]
        self._features = self.nn_forward_pass(input_dict)
        
        logits = self._logits(self._features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        """ Value function computation. """
        assert self._features is not None, "must call forward() first"
        
        # case for seperate value function network
        if self._value_branch_separate:
            return self.vf_output(self.vf_forward_pass()).squeeze(1)

        # case for shared value function network
        else:
            return self.vf_output(self._features).squeeze(1)
        
    def nn_forward_pass(self, input_dict):
        """ Forward pass for net. """
        # get observation
        obs = input_dict["obs"]
        x_inst = obs["inst_obs"]
        x_req = obs["req_obs"]
        
        # request embedding
        x_req_embed = self.iv_input(x_req)
        x_req_embed = self.iv_embed1(x_req_embed)
        x_req_embed = torch.sum(x_req_embed, axis=1) # concat all inputs
        x_req_embed = self.iv_embed2(x_req_embed)
    
        # concat instance features and request embedding
        x = torch.cat((x_inst, x_req_embed), 1)

        # pass through final layers
        x = self.concat_layer(x)
        return x
        
    def vf_forward_pass(self):
        """ Forward pass for vf if seperate net. """
        # get observation
        obs = self._last_in  #input_dict["obs"]
        x_inst = obs["inst_obs"]
        x_req = obs["req_obs"]
        
        # request embedding
        x_req_embed = self.vf_iv_input(x_req)
        x_req_embed = self.vf_iv_embed1(x_req_embed)
        x_req_embed = torch.sum(x_req_embed, axis=1) # concat all inputs
        x_req_embed = self.vf_iv_embed2(x_req_embed)
    
        # concat instance features and request embedding
        x = torch.cat((x_inst, x_req_embed), 1)

        # pass through final layers
        x = self.vf_concat_layer(x)
        return x