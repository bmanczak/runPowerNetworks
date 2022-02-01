import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import logging
import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict


from layers.graph_attention_layer import GATLayer, MultiHeadAttention, DecoderAttention
from models.utils import pool_per_substation, vectorize_obs, get_sub_adjacency_matrix
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


class SharedGraphModelling(nn.Module):

    def __init__(self, num_features:int, node_model_config:dict, substation_model_config:dict,
                sub_id_to_elem_id:dict, pool_method:str = "mean"):
        """
        Constructs the Encoder.

        Args:
            num_features (int): Number of features of each node.
            node_model_config (dict): Dictionary that specifies the node model.
                                      Has keys: hidden_dim, nheads, num_layers = 6, dropout=0
            substation_model_config (dict): Analogically to node_model_config.
            sub_id_to_elem_id (dict): Dictionary mapping substation id to element id.
            dropout (int, optional): Dropout probability. Defaults to 0.
        """
        super(SharedGraphModelling, self).__init__()
        self.linear = nn.Linear(num_features, node_model_config["hidden_dim"])
        self.node_model = StackedGATs(**node_model_config)
        self.substation_model = StackedGATs(**substation_model_config)

        self.sub_id_to_elem_id = sub_id_to_elem_id
        self.pool_method = pool_method

    def forward(self, x, adj_node, adj_substation):

        x = self.linear(x)
        #print(x.shape)
        x = self.node_model(x, adj_node)
        x = pool_per_substation(x, self.sub_id_to_elem_id, pooling_operator=self.pool_method)
        # print("pooled over substations!")
        # print("x.shape: ", x.shape)
        # print("adj_substation.shape: ", adj_substation.shape)
        x = self.substation_model(x, adj_substation)

        return x

class StackedGATs(nn.Module):

    def __init__(self, hidden_dim, nheads,
                num_layers = 2, dropout=0):
        """
        Stacks GAT layers.

        Args:
            num_features (int): Number of features of each node.
            hidden_dim ([int]): Number of features of each node after transformation.
            nheads (int): number of attention heads.
            num_layers (int, optional): Number of GAT layers. Defaults to 2.
            dropout (int, optional): Dropout probability. Defaults to 0.
        """
        super(StackedGATs, self).__init__()
        self.gat_layers = nn.ModuleList([
                GATLayer(hidden_dim, nheads, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x, adj):
        for layer in self.gat_layers:
            node_embeddings = layer(x, adj)

        return node_embeddings 

class ActorHead(nn.Module):

    def __init__(self, actor_head_config:dict):

        super(ActorHead, self).__init__()
        self.graph_model = StackedGATs(**actor_head_config)

        d_k = actor_head_config["hidden_dim"]//actor_head_config["nheads"] 
        self.get_context = MultiHeadAttention(n_head = actor_head_config["nheads"], d_model = actor_head_config["hidden_dim"],
                                            d_k = d_k, dropout= actor_head_config["dropout"],
                                            query_context = True)
        self.attention_prob =  DecoderAttention(temperature= np.power(d_k, 0.5))                       
        
    def forward(self, x, adj):
        x= self.graph_model(x, adj)
        # print("actor after graph model", x.shape)
        context = self.get_context(x,adj)
        # print("actor after context", context.shape)
        out = self.attention_prob(q =context, k = x, adj = adj)#.squeeze(1) [BATCH_SIZE, 1, NUM_ACTIONS]
        return out

class CriticHead(nn.Module):

    def __init__(self, critic_head_config:dict, concat_dim: int ):

        super(CriticHead, self).__init__()
        self.graph_model = StackedGATs(**critic_head_config)

        d_k = critic_head_config["hidden_dim"]//critic_head_config["nheads"] 
        self.get_context = MultiHeadAttention(n_head = critic_head_config["nheads"], d_model = critic_head_config["hidden_dim"],
                                            d_k = d_k, dropout= critic_head_config["dropout"],
                                            query_context = True)
        
        self.critizice =  nn.Linear(critic_head_config["hidden_dim"] + concat_dim, 1)                     
        
    def forward(self, x, actor_out, adj):
        x = self.graph_model(x, adj)
        context = self.get_context(x,adj)
        # print("critic after context", context.shape)
        # print("context", context.shape, "actor_out", actor_out.shape)
        context_and_actor = torch.cat((context, actor_out), dim = -1).squeeze(1)
        out = self.critizice(context_and_actor).squeeze(1) # [BATCH_SIZE]
        return out


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
        self.num_classes = 8

        self.node_model_config = model_config["custom_model_config"]["node_model_config"]
        self.substation_model_config = model_config["custom_model_config"]["substation_model_config"]
        self.actor_head_config = model_config["custom_model_config"]["actor_head_config"]
        self.critic_head_config = model_config["custom_model_config"]["critic_head_config"]

        self.mask_nodes = model_config["custom_model_config"].get("mask_nodes", None)
        self.pool_method = model_config["custom_model_config"].get("pool_method", "mean")

        self.sub_id_to_elem_id, self.topo_spec, \
        self.sub_id_to_action, self.line_to_sub_id = get_env_spec(model_config["custom_model_config"]["env_config"])
        self.cached_sub_adj = torch.from_numpy(get_sub_adjacency_matrix(self.line_to_sub_id))

        # Build the model
        self.shared = SharedGraphModelling(self.num_features, self.node_model_config, self.substation_model_config,
                                                self.sub_id_to_elem_id, pool_method=self.pool_method)
        
        self.actor = ActorHead(self.actor_head_config)
        self.critic = CriticHead(self.critic_head_config, concat_dim=self.num_classes)

    
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
        adj_node = input_dict["obs"]["connectivity_matrix"]
        self.batch_size = adj_node.shape[0]
        self.adj_substation = self.cached_sub_adj.repeat(self.batch_size, 1, 1).to(obs.device)
        
        # fill in the diagnal with 1 even when an element is disconnected
        #[adj[adj_num].fill_diagonal_(1) for adj_num in range(adj.shape[0])]

        self.shared_out = self.shared(obs, adj_node, self.adj_substation )
    
        actor_out = self.actor(self.shared_out, self.adj_substation ) # [BATCH_DIM, 1, NUM_SUBS]
        #print("Actor out", actor_out.shape)
        
        #disconnected_elements = input_dict["obs"]["topo_vect"] == -1
     

        logits = actor_out[:, 0, [0] + list(self.sub_id_to_action.keys())] # [BATCH_DIM, NUM_ACTIONS]

        self.logits = logits
        self._features = self.shared_out
        
        # print("look here")
        # print(torch.count_nonzero(obs))
        # print(obs)
        if not (logits == logits).all(): # torch.count_nonzero(obs).item() == 0: # dummy batch will produce nan -> change logits manually
            logits = torch.zeros_like(logits)
            logging.warning("Batch produced nan in logits, setting to 0")
            #assert (logits == logits).all(), "Probs should not contain any nans"

        return logits, state

    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        critic_out = self.critic(self._features, self.logits.unsqueeze(1), self.adj_substation) # [BATCH_DIM]]
        #print("Critic out", critic_out.shape)
        return critic_out


# class RllibSubsationModule(TorchModelV2, nn.Module):
#     def __init__(self, obs_space: gym.spaces.Space,
#                  action_space: gym.spaces.Space, num_outputs: int,
#                  model_config: ModelConfigDict, name: str):
#         """
#         Initialize the model.

#         Parameters:
#         ----------
#         obs_space: gym.spaces.Space
#             The observation space of the environment.
#         action_space: gym.spaces.Space
#             The action space of the environment.
#         num_outputs: int
#             The number of outputs of the model.

#         model_config: Dict
#             The configuration of the model as passed to the rlib trainer. 
#             Besides the rllib model parameters, should contain a sub-dict 
#             custom_model_config that stores the boolean for "use_parametric"
#             and "env_obs_name" for the name of the observation.
#         name: str
#             The name of the model captured in model_config["model_name"]
#         """

#         # Call the parent constructor.
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
#                               model_config, name)
#         nn.Module.__init__(self)
#         print("IN THAT INIT")
#         # Fetch the network specification
#         self.num_features = model_config["custom_model_config"]["num_features"]
#         self.hidden_dim = model_config["custom_model_config"].get("hidden_dim", 128)
#         self.nheads = model_config["custom_model_config"].get("nheads", 4)
#         self.num_layers = model_config["custom_model_config"].get("num_layers",3)
#         self.dropout = model_config["custom_model_config"].get("dropout", 0)
#         self.mask_nodes = model_config["custom_model_config"].get("mask_nodes", None)

#         self.sub_id_to_elem_id, self.topo_spec, \
#         self.sub_id_to_action, self.line_to_sub_id = get_env_spec(model_config["custom_model_config"]["env_config"])

#         self.cached_sub_adj = get_sub_adjacency_matrix(self.line_to_sub_id)

#         # Build the model
#         self.model = SubstationModel(self.num_features, self.hidden_dim, self.nheads, self.sub_id_to_elem_id, self.num_layers, self.dropout)

#         self._value_branch = nn.Linear(self.hidden_dim, 1)

#          # Holds the current "base" output (before logits layer).
#         self._features = None
#         # Holds the last input, in case value branch is separate.
#         self._last_flat_in = None


#     def forward(self, input_dict: Dict[str, TensorType],
#                 state: List[TensorType],
#                 seq_lens: TensorType):
#         # print("FORWARD!")
#         #print("input dic keys", input_dict.keys())
#        # print("obs_flat", input_dict["obs_flat"].shape)
#         #print("input dic obs", input_dict["obs"])
#         # print("input dic rho type", type(input_dict["obs"]["rho"]))
#         obs = vectorize_obs(input_dict["obs"], env_action_space = self.topo_spec)
#         adj = input_dict["obs"]["connectivity_matrix"]

#         # fill in the diagnal with 1 even when an element is disconnected
#         #[adj[adj_num].fill_diagonal_(1) for adj_num in range(adj.shape[0])]

#         disconnected_elements = input_dict["obs"]["topo_vect"] == -1
#         #print("Number of Disconnected elements", disconnected_elements.sum())
#         if disconnected_elements.any():
#             print("Disconnected elements: ", torch.argwhere(disconnected_elements==True))
        
#         adj = input_dict["obs"]["connectivity_matrix"]
#         #print("SHAPE OF ADJ", adj.shape)
#         non_zero_per_adj = torch.count_nonzero(adj, dim = [2]) # [BATCH_DIM]
#         if (non_zero_per_adj==0).any():
#             print("O japierfole!! Adding 1 to zero adjacency matrices")
#             print("The topo vect is", input_dict["obs"]["topo_vect"] )
#             print("Number of faulty adjacency matrices: ", adj[(non_zero_per_adj==0).any(1)].shape)
#             bad_adj = adj[(non_zero_per_adj==0).any(1)][0]
#             for row in range(bad_adj.shape[0]):
#                 if torch.sum(bad_adj[row, :])==0:
#                     #print("row", row, "is zero")
#                     print("row", row, bad_adj[row, :])

#         # print("Obs dim", obs.shape)
#         # print("Adjacency dim", input_dict["obs"]["connectivity_matrix"].shape)

#         substation_logits, node_embeddings, substation_embeddings = self.model(obs, input_dict["obs"]["connectivity_matrix"])
#         #self._last_flat_in = obs.reshape(obs.shape[0], -1)
#         #print("substation logits", substation_logits.shape)
#         #print("self.sub_id_to_action", self.sub_id_to_action)
#         #print("subsation_logits", substation_logits.shape, substation_logits)
#         #print("Subset actionable substations", list(self.sub_id_to_action.keys()))
#         substation_logits =  substation_logits[:, [0] + list(self.sub_id_to_action.keys())]
#         #print("Pruned substation logits", substation_logits.shape, substation_logits)
#         logits = substation_logits.squeeze(-1)
#         self._features = substation_embeddings
#         #print("LOGITS SHAPE", logits.shape)

#         return logits, state

#     def value_function(self) -> TensorType:
#         assert self._features is not None, "must call forward() first"
       
#         #print("Value branch shape",self._value_branch(torch.mean(self._features, dim = 1)).squeeze(1).shape )
#         return self._value_branch(torch.mean(self._features, dim = 1)).squeeze(1)





