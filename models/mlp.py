from typing import OrderedDict
import gym
import numpy as np

import torch 
import torch.nn as nn
import logging

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

# Limit values suitable for use as close to a -inf logit. These are useful
# since -inf / inf cause NaNs during backprop.
FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38


class SimpleMlp(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        """
        Initialize the model.

        Parameters:
        ----------
        obs_space: gym.spaces.Space
            The observation space of the environment.
        action_space: gym.spaces.Space
            The action space of the environment.
        num_outputs: int
            The number of outputs of the model.

        model_config: Dict
            The configuration of the model as passed to the rlib trainer. 
            Besides the rllib model parameters, should contain a sub-dict 
            custom_model_config that stores the boolean for "use_parametric"
            and "env_obs_name" for the name of the observation.
        name: str
            The name of the model captured in model_config["model_name"]
        """
        
        # Call the parent constructor.
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        
        # Fetch the network specification
        hiddens = list(model_config.get("fcnet_hiddens", [])) + \
            list(model_config.get("post_fcnet_hiddens", []))

        self.vf_share_layers = model_config.get("vf_share_layers")
        self.parametric_action_space = model_config["custom_model_config"].get("use_parametric", False)
        self.env_obs_name = model_config["custom_model_config"].get("env_obs_name", "grid")
      
        logging.info(f"Using parametric action space equals {self.parametric_action_space}")
       
        layers = []
        if self.parametric_action_space: # do not parametrize the action mask
            prev_layer_size = int(np.product(obs_space.shape) - action_space.n) # dim of the observation space
        else:
            prev_layer_size = int(np.product(obs_space.shape)) # dim of the observation space
        

        # Create hidden layers
        for size in hiddens:
            layers += [nn.Linear(prev_layer_size, size), nn.ReLU(inplace=True)]
            prev_layer_size = size
        
        
        self._logits = nn.Linear(prev_layer_size, num_outputs)
        self._hidden_layers = nn.Sequential(*layers)
        
        # Value function spec
        self._value_branch_separate = None
        if not self.vf_share_layers: # if we want to separate value function
            # Build a parallel set of hidden layers for the value net.
            if self.parametric_action_space: # do not parametrize the action mask
                prev_vf_layer_size = int(np.product(obs_space.shape) - action_space.n) # dim of the observation space
            else:
                prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers += [nn.Linear(prev_vf_layer_size, size), nn.ReLU(inplace=True)]
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = nn.Linear(prev_layer_size, 1)
  
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType):

        if self.parametric_action_space:
            obs = torch.concat(
                            [val for val in input_dict["obs"][self.env_obs_name].values()], dim=1) # [BATCH_DIM, obs_dim]
            #print("obs", obs.shape, obs)
            inf_mask = torch.clamp(torch.log(input_dict["obs"]["action_mask"]), FLOAT_MIN, FLOAT_MAX)
        else:
            if isinstance(input_dict["obs_flat"], OrderedDict):
                # logging.warning("applying custom flattening")
                # Flatten the dictionary and convert to a tensor
                obs = torch.cat(
                    [val for val in input_dict["obs_flat"].values()], dim = -1).float()
                if obs.ndim == 1: # edge case of batch size 1
                    obs = obs.unsqueeze(0)
            else:
                obs = input_dict["obs_flat"].float()

        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)

        logits = self._logits(self._features) 
        if self.parametric_action_space:
            logits += inf_mask
        #print("logits shape", logits.shape)

        if (torch.isnan(logits).any().item()) or (torch.isinf(logits).any().item()):
            logging.warning(f"Logits contain NaN values")
        return logits, state

    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)
