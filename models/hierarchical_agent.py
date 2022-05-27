import logging
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

from torchviz import make_dot
import visdom

from typing import Any, Dict
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

from layers.gat import GATModel, DecoderAttention
from layers.graph_attention_layer import GTrXL
from grid2op_env.grid_to_gym import Grid_Gym, get_env_spec
from models.utils import pool_per_substation, dense_to_edge_index,\
     tensor_to_data_list, sequence_mask, get_sub_adjacency_matrix, \
         get_elem_action_topo_map, vectorize_obs, logistic_func
from evaluation.restore_agent import restore_agent


FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38


class GreedySubModelNoWorker:
    def __init__(self, model, dist_class, env_config, num_to_sub):
        self.model = model
        self.dist_class = dist_class
        self.env_config = env_config
        self.num_to_sub = num_to_sub

    def choose_substation(self, obs):
        # print("Type obs", type(obs))
        # print("obs", obs)
        B = obs[list(obs.keys())[0]].shape[0]
        # Initialize all actions as 0 actions
        actions = torch.zeros(B).float() # batch size
        # Create a mask that selects the elements of a batch above rho_threshold
        query_model_mask = torch.where(obs["rho"].max(axis = 1).values > self.env_config["rho_threshold"], torch.ones(B), torch.zeros(B)) > 1e-6

        with torch.no_grad():
            logits, _ = self.model({"obs": obs, "obs_flat": torch.cat([val for val in obs.values()], dim = -1)})
            sampler = self.dist_class(logits, self.model)
            act_ix = sampler.sample().float()
            actions[query_model_mask] = act_ix[query_model_mask]
            # print("Act ix shape", act_ix.shape)

            # print("greedy_act_id: ", greedy_act_id)
            # print("trying compute actions on batch")
            # greedy_act_id = self.agent.compute_actions(obs)
            # print("Num to sub mapping", self.num_to_sub)
            act_on_sub = torch.tensor([self.num_to_sub[act_id.item()] for act_id in actions])
            return act_on_sub.unsqueeze(-1)


class GreedySubModel:

    def __init__(self, agent_path, checkpoint_num, trainer_type, agent = None, env_config = None):
        self.agent_path = agent_path
        self.checkpoint_num = checkpoint_num
        self.trainer_type = trainer_type

        print("agent path: ", self.agent_path)
        print("checkpoint num: ", self.checkpoint_num)
        print("trainer type: ", self.trainer_type)

        if agent is not None and env_config is not None:
            self.agent = agent
            self.env_config = env_config
        else:
            modify_keys_dict = {# convert all the training workers to evaluation workers
                                "num_workers": 0,
                                "evaluation_num_workers":2,
                                "evaluation_interval": 1}
                                
            self.agent, self.env_config = restore_agent(path = agent_path,
                                checkpoint_num=checkpoint_num,
                                trainer_type=trainer_type,
                                modify_keys= modify_keys_dict)
        self.rllib_env = Grid_Gym(self.env_config)

        self.actionable_substaions = sorted(self.rllib_env.env_gym.agent.sub_id_to_action.keys())
        self.num_to_sub =self.rllib_env.env_gym.agent.num_to_sub

        print("Model initialized!")
    def choose_substation(self, obs):

        if max(obs["rho"]) < self.env_config["rho_threshold"]:
            return 0 # do nothing
        else:
            greedy_act_id = self.agent.compute_single_action(obs)
            
            print("greedy_act_id: ", greedy_act_id)
            # print("trying compute actions on batch")
            # greedy_act_id = self.agent.compute_actions(obs)
            act_on_sub = self.num_to_sub[greedy_act_id]
            if isinstance(act_on_sub, int):
                act_on_sub = torch.from_numpy(np.array([act_on_sub]))
            print("act_on_sub: ", act_on_sub)

        return act_on_sub

class HierarchicalGraphModel(nn.Module):

    def __init__(self, node_model_config, sub_model_config,
                sub_id_to_elem_id, sub_id_to_action,pretrained_substation_model = None, use_gtrxl = False):
        super(HierarchicalGraphModel, self).__init__()

        self.node_model_config = node_model_config
        self.sub_model_config = sub_model_config
        # self.decoder_model_config = decoder_model_config
        self.pretrained_substation_model = pretrained_substation_model
        self.use_gtrxl = use_gtrxl

        # Sorting important to preserve order and fetch correct nodes in the node module
        self.sub_id_to_elem_id = {k:sorted(v) for k,v in sub_id_to_elem_id.items()} # sort the elements
        # Increment the substation id by 1 to allow the 0th substation to be a do-nothing 
        # self.sub_id_to_elem_id_and_dn = {i+1:k for i,k in self.sub_id_to_elem_id.items()}
        # self.num_to_sub[0] = [] # 

        self.sub_id_to_action = sub_id_to_action
        
        if self.use_gtrxl:
            self.node_model = GTrXL(**self.node_model_config)
        else:
            self.node_model = GATModel(**node_model_config)
        

        # if self.pretrained_model_config is not None:
        self.choose_sub_ln_key = "hidden_dim" if self.use_gtrxl else "c_out"
        if self.pretrained_substation_model is not None:
            logging.info("Using pretrained model for choosing the substation.")
            self.use_sub_pretrained = True #self.pretrained_model_config.get("pretrained", False)       
        else:
        
            self.choose_substation_ln = nn.Linear(sub_model_config[self.choose_sub_ln_key], 1)
            self.use_sub_pretrained = False
        sub_model_config.pop("pretrained_model_config", None) # delete pretrained flag

        if use_gtrxl:
            self.sub_model = GTrXL(**self.sub_model_config)
        else:
            self.sub_model = GATModel(**sub_model_config)
        print("Substation model: ", self.sub_model)

       # self.choosable_subs = [0] + list(sub_id_to_action.keys()) # used for masked fill
        self.choosable_subs = list(sub_id_to_action.keys()) # used for masked fill

        print("choosable subs: ", self.choosable_subs)
        print("list(sub_id_to_action.keys())", list(sub_id_to_action.keys()))
        

    def node_sub_forward(self,obs, node_adj, sub_adj):
        """
        Performs the forward pass of the node and substation models,
        including the pulling operation over substations.
        """
        B = obs.shape[0]
        # print("Batch size", B)
        
        if self.use_gtrxl:
            node_embeddings = self.node_model(obs, node_adj)
        else: # use PyTorch Geoemtric GAT
            batch = tensor_to_data_list(obs, node_adj)
            node_x, edge_index_node = batch.x, batch.edge_index
            node_embeddings = self.node_model(node_x, edge_index_node)
            
        pooled_embeddings = pool_per_substation(node_embeddings.reshape(B, 56,self.node_model_config[self.choose_sub_ln_key]), self.sub_id_to_elem_id)

        if self.use_gtrxl:
            substation_embeddings = self.sub_model(pooled_embeddings, sub_adj)
        else:# Get data back to PyTorch geoemtric format
            substation_batch = tensor_to_data_list(pooled_embeddings, sub_adj)
            sub_x, edge_index_sub = substation_batch.x, substation_batch.edge_index
            substation_embeddings = self.sub_model(sub_x, edge_index_sub)

        return node_embeddings, substation_embeddings

    def update_obs(self, obs:torch.Tensor, chosen_substation: torch.Tensor):

        """
        Given a substation to intervene on changes the "predict_config" part of the observation.
        """

        updated_obs = obs.clone()
        #print("Obs clone shape", obs.shape)
        # count_elem_change = 0
        #print("chosen_substation: ", chosen_substation)
        for i, chosen_sub in enumerate(chosen_substation):
            # print("chosen_sub: ", chosen_sub.shape, chosen_sub)

            #count_elem_change += len(self.sub_id_to_elem_id[chosen_sub.item()])
            assert (updated_obs[i, self.sub_id_to_elem_id[chosen_sub.item()], 3] ==0).all(), \
                "The being_modified' part of the observation should be 0"
            updated_obs[i, self.sub_id_to_elem_id[chosen_sub.item()], 3] = 1
           # print("Updating the obs index for substation: ", self.sub_id_to_elem_id[chosen_sub.item()])

        # print("count_elem_change", count_elem_change)
        # print("Updated obs - obs sum", (updated_obs.sum() - obs.sum()).item())
        # print("Sub station choice 0", chosen_substation[0])
        # print("First observation", updated_obs[0])
        #assert ((updated_obs.sum() - obs.sum()).item() ==  count_elem_change), "Observation has not been modified correctly"

        return updated_obs

    def choose_substation(self, substation_embeddings:torch.Tensor):
        """
        Given substation embeddings returns the logits of the substations.
        """
        substation_logits = self.choose_substation_ln(substation_embeddings) # [BATCH_DIM, NUM_SUBS, 1]
        print("substation logits", substation_logits.shape)
        print("Substation logit 0", substation_logits[0])
        mask = torch.zeros_like(substation_logits)
        mask[:, self.choosable_subs, :] = 1
        substation_logits = substation_logits.masked_fill_(mask == 0, FLOAT_MIN)

        return substation_logits

    
    def get_sub_choice(self, obs, node_adj, sub_adj, obs_non_vectorized = None):
        """
        Runs the forward method 
        """

        if self.use_sub_pretrained:
            assert self.use_sub_pretrained, "Pretrained substation model flag is not set but model is being used."
            assert obs_non_vectorized is not None, "Pretrained model requieres non-vectorized observations."
            if self.pretrained_substation_model.env_config.get("conn_matrix", False) == False:
                obs_non_vectorized.pop("connectivity_matrix", None)
            with torch.no_grad():
                sub_choice = self.pretrained_substation_model.choose_substation(obs_non_vectorized)

            return sub_choice
        else:
            B = obs.shape[0]

            node_embeddings_1, substation_embeddings_1 = self.node_sub_forward(obs, node_adj, sub_adj)
            sub_logits = self.choose_substation(substation_embeddings_1.reshape(B, 14,self.sub_model_config[self.choose_sub_ln_key]))
            sub_choice = torch.argmax(F.softmax(sub_logits, dim = 1), dim = 1)

            return sub_choice, node_embeddings_1, substation_embeddings_1

    
    def forward(self, obs, node_adj, sub_adj, obs_non_vectorized = None):
        
        B = obs.shape[0]

        if self.use_sub_pretrained:
           
            sub_choice = self.get_sub_choice(obs, node_adj, sub_adj, obs_non_vectorized)
            # print("Using pretrained mode sub choice",sub_choice)
        else:
            # print("Learning model")
            sub_choice, node_embeddings_1, substation_embeddings_1 = self.get_sub_choice(obs, node_adj, sub_adj)

        updated_obs = obs # self.update_obs(obs, sub_choice)
        node_embeddings_2, substation_embeddings_2 = self.node_sub_forward(updated_obs, node_adj, sub_adj)

        if not self.use_sub_pretrained:
            node_embeddings_2 += node_embeddings_1
            substation_embeddings_2 += substation_embeddings_1

        return node_embeddings_2.reshape(B, 56,self.node_model_config[self.choose_sub_ln_key]), \
                substation_embeddings_2.reshape(B, 14,self.sub_model_config[self.choose_sub_ln_key]), \
                sub_choice
                


class Action_Decoder(nn.Module):

    def __init__(self, decoder_model_config, sub_id_to_elem_id, element_to_action_num,\
            action_to_topology,node_model_config, sub_model_config, obs_dim, act_dim ):
        super(Action_Decoder, self).__init__()

        self.choose_sub_ln_key = "hidden_dim" if sub_model_config.get("hidden_dim", None) else "c_out"
        self.decoder_model_config = decoder_model_config
        self.org_obs_projection = nn.Linear(obs_dim, node_model_config[self.choose_sub_ln_key])

        # Decide on the type of the decoder: GNN with node level classifier or 
        # the attenton based decoding as in the paper "Attention learn to solve routing problems"
        self.routing_decoder = self.decoder_model_config.get("routing_decoder", False)

        # self.decoder_model_config["d_out"] = decoder_model_config["d_out"] + 128 # for the extra observation 
        # self.decoder_model_config["d_q"] = decoder_model_config["d_q"] + 128 # for the extra observation 
        # self.decoder = DecoderAttention(**self.decoder_model_config)
        decoder_model_config["c_in"] = 3*node_model_config[self.choose_sub_ln_key] # concat node embeddings, substation embeddings and the projected observation
        print("Decoder model config", decoder_model_config)
        self.node_classifier = GATModel(**decoder_model_config)

        self.act_dim = act_dim
        self.sub_id_to_elem_id = sub_id_to_elem_id
        self.element_to_action_num = element_to_action_num
        self.action_to_topology = action_to_topology

        
        self.node_model_config = node_model_config
        self.sub_model_config = sub_model_config
        

    def forward(self, org_obs, node_embeddings, substation_embeddings, sub_choice):

        obs_representation = self.org_obs_projection(org_obs)
        # print("Obs representation", obs_representation.shape)
        config_nodes, config_sub = self.fetch_node_and_sub_embeddings(node_embeddings,
                                         substation_embeddings,
                                        sub_choice)

        # if self.decoder_model_config.get("routing_decoding", False):
        # padded_config_nodes = pad_sequence(config_nodes, batch_first=True)
        # mask = sequence_mask(seq = padded_config_nodes.clone().detach(), padding_idx = 0)[:, :, 0] # [BACTH SIZE, #MAX_NODES]
        
        # query = torch.concat([obs_representation.unsqueeze(1), config_sub], dim = -1)

        # busbar_one_logits = self.decoder(q = query, k = padded_config_nodes, mask = mask)
        data_lst = []
        for batch_num, config_node in enumerate(config_nodes):
            num_nodes = config_node.shape[0]
            # print("Obs representation shape", obs_representation[batch_num].unsqueeze(0).repeat(num_nodes, 1).shape)
            # print("Config sub shape", config_sub[batch_num].repeat(num_nodes,1).shape)
            # print("Config node shape", config_node.shape)
            x_enriched = torch.concat([obs_representation[batch_num].unsqueeze(0).repeat(num_nodes, 1),
                            config_sub[batch_num].repeat(num_nodes,1), config_node], dim = -1)
            # x_enriched =config_node
            # print("X enriched shape", x_enriched.shape)
            
            # data_lst.append(Data(x = x_enriched, edge_index = dense_to_edge_index(torch.ones((num_nodes, num_nodes)))))
            data_lst.append(Data(x = x_enriched, edge_index = 
             dense_to_edge_index(torch.ones_like(torch.ones((num_nodes, num_nodes)) ))
            ))

        batch_to_classify = Batch.from_data_list(data_lst)
        node_preds = self.node_classifier(batch_to_classify.x, batch_to_classify.edge_index)
        print("THESE NODE PREDS", node_preds)

        # Getting predictions back to a list of size B
        node_preds_lst = []
        ix = 0
        for batch_num, config_node in enumerate(config_nodes):
            node_preds_lst.append(node_preds[ix:ix+config_node.shape[0]])
            ix += config_node.shape[0]

        assert sum([len(elem) for elem in node_preds_lst])==node_preds.shape[0],\
                f"Not all nodes have predictions. Should have f{node_preds.shape[0]} predictions but \
                    there are {sum([len(elem) for elem in node_preds_lst])} predictions"

        return node_preds_lst, sub_choice


    def fetch_node_and_sub_embeddings(self, node_embeddings, substation_embeddings, sub_choice):
        """
        Given the second forward pass of the network fetches the node and substaion embeddings
        chosen by the higher level policy.
        """
        assert node_embeddings.ndim == 3, "Node embeddings required to be in the format [BATCH_DIM, #NODE, HIDDEN_DIM]"
        assert substation_embeddings.ndim == 3, "Substation embeddings required to be in the format [BATCH_DIM, #SUB, HIDDEN_DIM]"

        # lst_form_batch = []
        tensor_lst = []
        B = sub_choice.shape[0] # batch size

        # Get the element embeddings belonging to the chosen substation
        for i in range(B):
            ix_elems_sub = self.sub_id_to_elem_id[sub_choice[i].item()]
            tensor_lst.append(node_embeddings[i, ix_elems_sub, : ])
            
        # Get the embeddings of the chosen substations
        sub_choice_embeddings = torch.gather(input = substation_embeddings, dim = 1,
                                    index = sub_choice.unsqueeze(-1).repeat(1,1,self.sub_model_config[self.choose_sub_ln_key])) # -1 ==self.sub_model_config["c_out"]
        # print("Sub choice embeddings shape", sub_choice_embeddings.shape)
        # print("Sub choice shape", sub_choice.shape)

        # print(sub_choice_embeddings[0, 0, :])
        assert torch.abs((substation_embeddings[0, sub_choice[0,0].item(), :]-sub_choice_embeddings[0, 0, :])).sum() < 1e-6, "Sub choice embeddings are incorrect"
        return tensor_lst, sub_choice_embeddings

    def decode_to_probs(self,busbar_probs, sub_choice):
        
        # B = busbar_probs.shape[0] # batch size
        B = len(busbar_probs) # batch size
        
        # convert to probabilities
        # busbar_one_probs = torch.softmax(busbar_probs, dim = -1) #1 /  (1 + torch.exp(-busbar_one_logits))
        busbar_one_probs = busbar_probs
        # print("Busbar one probs shape", busbar_one_probs)
        # Get the elements of the chosen substation
        switchable_elem_subs = [self.sub_id_to_elem_id[single_sub_choice.item()] for single_sub_choice in sub_choice] 
        # Get the action int for the chosen substation
        actions_for_topo = [[action for action, bus in self.element_to_action_num[single_switchable_elem_sub[0]]] for single_switchable_elem_sub in switchable_elem_subs]# we can use 0 because each element belonging to the same substaion has the same actions
  
        final_action_distr_logits = torch.full((B,self.act_dim), FLOAT_MIN) # [1, NUM_ACTIONS]
        # print("actions_for_topo", actions_for_topo)
        for num_in_batch, one_actions_for_topo in enumerate(actions_for_topo):
            actions_sub = np.array([self.action_to_topology[action_topo] for action_topo in one_actions_for_topo])
            num_modifiable_elements = actions_sub.shape[1] # needed to get rid of the padded values
            
            # print("Actions_sub", actions_sub)
            # print("Num_modifiable_elements", num_modifiable_elements)
            probs_bus_bar_one = logistic_func(busbar_one_probs[num_in_batch]).flatten() # proper p
            # print("Probs bus bar one", probs_bus_bar_one)
            probs_bus_bar_two = 1 - probs_bus_bar_one
            probs_bus_one_bus_two = torch.stack( 
                [torch.log(probs_bus_bar_one), # prob of the elements being on busbar 1
                torch.log(probs_bus_bar_two)] # probs of the elements being on busbar 2
                ) # [2, NUM_ELEMENTS] where 2 stands for the two buses
            # print("Shape Probs bus one bus two", probs_bus_one_bus_two.shape)
            # print("Probs bus one bus two", probs_bus_one_bus_two)
            # probs_bus_one_bus_two = busbar_probs[num_in_batch].T
            # print("Transposed probs_bus_one_bus_two", probs_bus_one_bus_two.T.shape)

            actions_bus_tensor = torch.from_numpy(actions_sub)[:,:,1]
            actions_bus_ix_tensor = actions_bus_tensor - 1 # bus 1 at ix 0, bus 2 at ix 2

            num_actions_at_sub = actions_bus_ix_tensor.shape[0]
            probs_to_choose = probs_bus_one_bus_two.T.unsqueeze(0).repeat(num_actions_at_sub,1,1) # [NUM_ACTIONS, NUM_ELEMENTS, 2]
            probs_per_node = torch.gather(input = probs_to_choose, dim = -1, index = actions_bus_ix_tensor.unsqueeze(-1))
            # PROD when probs, SUM when LOG PROBS 
            probs_per_available_action = torch.sum(probs_per_node, dim = 1).squeeze(1)
            # final_action_distr_logits[num_in_batch, one_actions_for_topo] = torch.log(probs_per_available_action)
            final_action_distr_logits[num_in_batch, one_actions_for_topo] = probs_per_available_action
            
            # Get the normalzied per action probability
            # F.softmax(final_action_distr_logits[0])
        # print("Sub choice", sub_choice)
        # print("Action for topo", actions_for_topo)
        # print("Final action distr logits", F.softmax(final_action_distr_logits, dim = -1))
        return final_action_distr_logits
            
        

class HierarchicalAgent(TorchModelV2, nn.Module):
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
        print("MODEL CONFIG IS")
        print(model_config)
        print("-------------------------------")

        self.obs_dim = gym.spaces.utils.flatdim(obs_space)
        self.act_dim = gym.spaces.utils.flatdim(action_space)
        print("Obs dim", self.obs_dim)
        print("Act dim", self.act_dim)
        # self.act_dim[1,2,3] 

        self.node_model_config = model_config["custom_model_config"]["node_model_config"]
        self.sub_model_config = model_config["custom_model_config"]["sub_model_config"]
        self.pretrained_model= model_config["custom_model_config"].get("pretrained_substation_model", None)
        self.use_gtrxl = model_config["custom_model_config"].get("use_gtrxl", False)
        self.choose_sub_ln_key = "hidden_dim" if self.use_gtrxl else "c_out"
        
        self.decoder_model_config = model_config["custom_model_config"]["decoder_model_config"]
        
        # Get environment information needed for modelling
        self.sub_id_to_elem_id, self.topo_spec, \
        self.sub_id_to_action, self.line_to_sub_id = get_env_spec(model_config["custom_model_config"]["env_config"])
        self.rllib_env =  Grid_Gym(model_config["custom_model_config"]["env_config"]); # copy of the training env to get the elem_topo_action_mappings
        self.element_to_action_num, self.action_to_topology = get_elem_action_topo_map(self.rllib_env)
        # Get the adjacency matrices and keep them fixed during training
        self.cached_sub_adj = torch.from_numpy(pickle.load(open(\
           model_config["custom_model_config"]["sub_adj_mat_path"]
            , "rb"))).unsqueeze(0)
        self.cached_node_adj = torch.from_numpy(pickle.load(open(\
           model_config["custom_model_config"]["node_adj_mat_path"]
            , "rb"))).unsqueeze(0)
        assert self.cached_sub_adj.sum().item() > 0, "Substation adjacency matrix is empty"
        assert self.cached_node_adj.sum().item() > 0, "Node adjacency matrix is empty"
        

        # Build the models
        self.node_sub_actor = HierarchicalGraphModel(self.node_model_config, self.sub_model_config,
                self.sub_id_to_elem_id, self.sub_id_to_action,pretrained_substation_model = self.pretrained_model, use_gtrxl = self.use_gtrxl)
        
        self.node_sub_critic = HierarchicalGraphModel(self.node_model_config, self.sub_model_config,
                self.sub_id_to_elem_id, self.sub_id_to_action,pretrained_substation_model = self.pretrained_model, use_gtrxl = self.use_gtrxl)

        self.actor_head = Action_Decoder(decoder_model_config=self.decoder_model_config, 
                                        sub_id_to_elem_id=self.sub_id_to_elem_id,
                                        element_to_action_num=self.element_to_action_num, action_to_topology=self.action_to_topology,
                                        sub_model_config = self.sub_model_config, node_model_config = self.node_model_config,
                                        obs_dim= self.obs_dim, act_dim=self.act_dim)
        
        # For the value function downscale substation_embeddings to 32 then concat
        self.value_fn_downscale = nn.Linear(self.sub_model_config[self.choose_sub_ln_key] , 32)
        self.value_fn_classify = nn.Linear(32*14, 1) # concat of 14 substaions
        self.value_branch = nn.Sequential(
          nn.Linear(self.obs_dim, 256), nn.ReLU(inplace=True),
          nn.Linear(256,256), nn.ReLU(inplace=True),
          nn.Linear(256,256),nn.ReLU(inplace=True),
           nn.Linear(256,1)
        )  

    
         # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None
        self.cur_val = None
        self.fwd_iters = 0
        self.grad_dots: Dict[str, Any] = {}
        self.visdom = visdom.Visdom()

    def export_dots(self):
        if self.training:
            for k, v in self.grad_dots.items():
                path = f"/Users/blazejmanczak/Desktop/School/Year2/Thesis/runPowerNetworks/log_files/my_grads/{k}.png"
                v.render(path, format="png")
    def add_grad_dot(self, tensor, name):
        if  self.training:
            self.grad_dots[f"{name}_iter_{self.fwd_iters}"] = make_dot(
                tensor.clone(), params=dict(list(self.named_parameters()))
            )
    
    def get_num_comp_graph_nodes(self):
        """Prints the number of nodes in the torch computational graph. Use
        this to ensure we don't leak gradients from previous passes"""
        if self.training:
            for k, v in self.grad_dots.items():
                self.visdom_mets["line"][
                    k.split("_iter")[0] + "_comp_graph_nodes"
                ] = np.array([len(v.body)])

    def is_dummy_batch(self,obs_flat):
        """
        Check if the observation comes from a dummy batch,
        i.e. all observations equal to 0.
        """
        return torch.all(torch.eq(obs_flat, 0))
    
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType):

        # Create a value 
        self.obs = vectorize_obs(input_dict["obs"], env_action_space = self.topo_spec)
        self.obs_non_vectorized = input_dict["obs"]
        self.obs_flat = input_dict["obs_flat"]

        self.batch_size = input_dict["obs_flat"].shape[0]
        self.adj_node = self.cached_node_adj.repeat(self.batch_size, 1, 1).to(self.obs.device)
        self.adj_substation = self.cached_sub_adj.repeat(self.batch_size, 1, 1).to(self.obs.device)
        
        # Actor  
        self.node_embeddings_actor, self.substation_embeddings_actor, self.sub_choice = self.node_sub_actor(self.obs, self.adj_node \
                                                , self.adj_substation, self.obs_non_vectorized)
        busbar_one_logits, sub_choice = self.actor_head(input_dict["obs_flat"], self.node_embeddings_actor, self.substation_embeddings_actor, self.sub_choice)
        logits = self.actor_head.decode_to_probs(busbar_one_logits, sub_choice) # [BATCH_DIM, NUM_ACTIONS]

        self.logits = logits

        # print("node embedding shape", self.node_embeddings_actor.shape) 
        # print("substation embedding shape", self.substation_embeddings_actor.shape)
        # print("sub_choice shape", self.sub_choice.shape)
        # print("logits shape", self.logits.shape)

        if not (logits == logits).all(): # torch.count_nonzero(obs).item() == 0: # dummy batch will produce nan -> change logits manually
            faulty_batches = np.argwhere((logits == logits).all(1) == False)
            print("None detected in the following batches", faulty_batches.shape[0])
            for faulty_obs in faulty_batches:
                print("Faulty obs", faulty_obs)
                print("Sub choice", self.sub_choice[faulty_obs])
                print("Busbar one logits", busbar_one_logits[faulty_obs])
                print("Logits", self.logits[faulty_obs])
                
            logits = torch.zeros_like(logits)
            logging.warning("Batch produced nan in logits, setting to 0")
            #assert (logits == logits).all(), "Probs should not contain any nans"

        # print("End forward")
        # print("-"*40)

        # self.add_grad_dot(logits, "logits")
        # self.add_grad_dot(self.node_embeddings_actor, "node_embeddings")
        # self.add_grad_dot(self.substation_embeddings_actor, "substation_embeddings")
        # self.fwd_iters += 1
        # self.export_dots()
        # # self.get_num_comp_graph_nodes()
        # if self.training:
        #     self.grad_dots.clear()
        # print("MODEL WEIGHTS", self.actor_head.layer[0].weight)
        return logits, state

    def value_function(self) -> TensorType:
        assert self.obs is not None, "must call forward() first"
        # _, self.substation_embeddings_critic, _ = self.node_sub_actor(self.obs, self.adj_node, self.adj_substation, self.obs_non_vectorized)
        # self.substation_embedding_critic = self.substation_embeddings_critic + self.substation_embeddings_actor
        # down_concat = self.value_fn_downscale(self.substation_embedding_critic).reshape(self.batch_size, -1)
        # critic_out = self.value_fn_classify(down_concat).squeeze(-1)
        critic_out = self.value_branch(self.obs_flat)
        # print("End critic")
        # print("-"*40)
        return critic_out.squeeze(-1)