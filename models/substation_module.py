import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import logging

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict


from layers.graph_attention_layer import GATLayer
from models.utils import pool_per_substation, vectorize_obs
from grid2op_env.grid_to_gym import get_env_spec
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

torch.autograd.set_detect_anomaly(True)
class SubstationModel(nn.Module):
    def __init__(self, num_features, hidden_dim, nheads, sub_id_to_elem_id,
                num_layers = 6, dropout=0):
        """
        Constructs the Encoder.

        Args:
            num_features (int): Number of features of each node.
            hidden_dim ([int]): Number of features of each node after transformation.
            nheads (int): number of attention heads.
            num_layers (int, optional): Number of GAT layers. Defaults to 6.
            dropout (int, optional): Dropout probability. Defaults to 0.
        """
        super(SubstationModel, self).__init__()
        self.linear = nn.Linear(num_features, hidden_dim)
        self.gat_layers = nn.ModuleList([
                GATLayer(hidden_dim, nheads, dropout=dropout) for _ in range(num_layers)])

        self.classify_substations = nn.Linear(hidden_dim, 1) # a binary classifier on each substation
        self.sub_id_to_elem_id = sub_id_to_elem_id
        

    def forward(self, x, adj):

        x = self.linear(x)
        for layer in self.gat_layers:
            node_embeddings = layer(x, adj)
        substation_embeddings = pool_per_substation(node_embeddings, self.sub_id_to_elem_id)
        
        # substation_prob_distr = F.softmax(self.classify_substations(substation_embeddings), dim = 1)
        substation_logits = self.classify_substations(substation_embeddings) # logits

        return substation_logits, node_embeddings, substation_embeddings

class RllibSubsationModule(TorchModelV2, nn.Module):
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
        print("IN THAT INIT")
        # Fetch the network specification
        self.num_features = model_config["custom_model_config"]["num_features"]
        self.hidden_dim = model_config["custom_model_config"].get("hidden_dim", 128)
        self.nheads = model_config["custom_model_config"].get("nheads", 4)
        self.num_layers = model_config["custom_model_config"].get("num_layers",3)
        self.dropout = model_config["custom_model_config"].get("dropout", 0)
        self.mask_nodes = model_config["custom_model_config"].get("mask_nodes", None)

        self.sub_id_to_elem_id, self.topo_spec, self.sub_id_to_action = get_env_spec(model_config["custom_model_config"]["env_config"])
        # Build the model
        self.model = SubstationModel(self.num_features, self.hidden_dim, self.nheads, self.sub_id_to_elem_id, self.num_layers, self.dropout)
        self.project_to_8 = nn.Linear(14,8)

        self._value_branch = nn.Linear(self.hidden_dim, 1)

         # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None


    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType):
        # print("FORWARD!")
        #print("input dic keys", input_dict.keys())
       # print("obs_flat", input_dict["obs_flat"].shape)
        #print("input dic obs", input_dict["obs"])
        # print("input dic rho type", type(input_dict["obs"]["rho"]))
        obs = vectorize_obs(input_dict["obs"], env_action_space = self.topo_spec)
        # print("Obs dim", obs.shape)
        # print("Adjacency dim", input_dict["obs"]["connectivity_matrix"].shape)

        substation_logits, node_embeddings, substation_embeddings = self.model(obs, input_dict["obs"]["connectivity_matrix"])
        #self._last_flat_in = obs.reshape(obs.shape[0], -1)
        #print("substation logits", substation_logits.shape)
        #print("self.sub_id_to_action", self.sub_id_to_action)
        #print("subsation_logits", substation_logits.shape, substation_logits)
        #print("Subset actionable substations", list(self.sub_id_to_action.keys()))
        substation_logits =  substation_logits[:, [0] + list(self.sub_id_to_action.keys())]
        #print("Pruned substation logits", substation_logits.shape, substation_logits)
        logits = substation_logits.squeeze(-1)
        self._features = substation_embeddings
        #print("LOGITS SHAPE", logits.shape)

        return logits, state

    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
       
        #print("Value branch shape",self._value_branch(torch.mean(self._features, dim = 1)).squeeze(1).shape )
        return self._value_branch(torch.mean(self._features, dim = 1)).squeeze(1)





