import torch 
import numpy as np
import torch_geometric
import grid2op

from torch_geometric.data import Data, Batch
from sknetwork.utils import edgelist2adjacency
from typing import Tuple, List, Dict
from collections import defaultdict

FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38

def vectorize_obs(obs, env_action_space, hazard_threshold = 0.9):
    """
    Vectorize the gym observation.

    :param OrderedDict obs: gym observation
    :param Grid2Op_environment env: grid2op environment. Used to fetch the 
        ids of different objects.
    :param float hazard_threshold
    """

    length = env_action_space.dim_topo # number of bus bars == number of nodes in the graph
    batch_size = 1 if obs["rho"].ndim == 1 else obs["rho"].shape[0]
    # print("batch size", batch_size)
    # for all keys in obs print the value shape
    # for key in obs.keys():
    #     print(key, obs[key].shape)
    # rho is symmetric for both ends of the line [batch_dim, 56,1]
    for obs_type in obs.keys():
        if isinstance(obs[obs_type], np.ndarray):
            obs[obs_type] = torch.from_numpy(obs[obs_type])
    device = obs["rho"].device
    rho = torch.zeros((batch_size, length), device = device)
    rho[:,env_action_space.line_or_pos_topo_vect] = obs["rho"]
    rho[:, env_action_space.line_ex_pos_topo_vect] = obs["rho"]

    # active power p [batch_dim, 56,1]
    p = torch.zeros((batch_size, length), device = device)
    p[:,env_action_space.gen_pos_topo_vect] = obs["gen_p"]# generator active production
    p[:,env_action_space.load_pos_topo_vect] = obs["load_p"] # load active consumption
    p[:,env_action_space.line_or_pos_topo_vect] = obs["p_or"] # origin active flow
    p[:,env_action_space.line_ex_pos_topo_vect] = obs["p_ex"] # Extremity active flow

    # overflow [batch_dim, 56,1]
    over = torch.zeros((batch_size, length), device = device)
    over[:,env_action_space.line_or_pos_topo_vect] = obs["timestep_overflow"].float()
    over[:,env_action_space.line_ex_pos_topo_vect] = obs["timestep_overflow"].float()

    # one-hot topo vector [batch_dim, 56,3]
    topo_vect_one_hot = torch.zeros((batch_size, length,3), device = device)
    topo_vect = obs["topo_vect"].unsqueeze(0).repeat(batch_size,1) if obs["topo_vect"].ndim != 2 else obs["topo_vect"] # [batch_dim, 56,1]
    # print("obs_topo after",obs["topo_vect"].shape)
    # topo_vect = obs["topo_vect"] # [batch_dim, 56,1]
    topo_vect[topo_vect==-1] = 0 # change disconneted from -1 to 0
    topo_vect_one_hot = torch.nn.functional.one_hot(topo_vect.to(torch.int64), num_classes=3)
    # print("topo_vect_one_hot", topo_vect_one_hot.shape)

    # powerline maintenance
    # maintenance = torch.zeros((batch_size, length), device = device)
    # maintenance[env_action_space.line_or_pos_topo_vect] = obs["maintenance"]).float()
    # maintenance[env_action_space.line_ex_pos_topo_vect] = obs["maintenance"]).float()

    # manual feature thresholding 
    # is being modified 
    predict_config = torch.zeros((batch_size, length), device = device)

    # print("Shape of rho", rho.shape)
    # print("Shape of p", p.shape)
    # print("Shape of over", over.shape)
    # print("Shape of topo_vect_one_hot", topo_vect_one_hot.shape)
    vectorized_obs = torch.stack([rho,p,over, predict_config], dim = -1)
    # print("Shape of vectorized_obs", vectorized_obs.shape)
    # print("vectorized_obs", vectorized_obs.shape)
    vectorized_obs = torch.concat([vectorized_obs, topo_vect_one_hot], dim = -1)

    return vectorized_obs

def logistic_func(x):
    return 1/(1+torch.exp(-x))

def pool_per_substation(arr, sub_to_id, pooling_operator = "mean"):
    """
    Pool the observations over the substations.
    
    Keyword arguments:
    ----------
    arr: np.array
        The array to pool over
    sub_to_id: dict
        The dictionary that maps the substation id to the index of the array.
    pooling_operator: str
        "mean" or "max". Defaults to mean.
    """

    pooled = [] 
    for sub, elements in sub_to_id.items():
        if pooling_operator == "max":
            pooled.append(torch.mean(arr[:, elements, :], dim = 1, keepdim=True))
        else:
            pooled.append(torch.mean(arr[:, elements, :], dim = 1, keepdim=True))
    
    return torch.cat(pooled, dim = 1)

def get_sub_adjacency_matrix(line_to_sub_id_tuple: Tuple[List, List], rho: np.array = None,
                             add_self_loops: bool = True) -> np.array:
    """
    Get the adjacency matrix of the substations taking into
    account the disabled line.
    
    Keyword arguments:
    ----------
    line_to_sub_id_tuple: tuple of lists
        The list of edges of the graph.
    rho: list
        The list of the rho values.
    """
    edge_list = np.array(list(zip(*line_to_sub_id_tuple)))

    if rho is not None:
        working_lines = np.argwhere(rho>0).flatten()
        edge_list = edge_list[working_lines]

    adj = edgelist2adjacency(edge_list, undirected=True).todense()

    if add_self_loops:
        np.fill_diagonal(adj, 1)

    return adj
    

def dense_to_edge_index(adj: torch.tensor):
    """
    Transform a dense adjacency matrix to edge index.

    Parameters:
    adj: torch.tensor,
        Adjacencey matrix with entries [n_nodes, n_nodes]

    """
    # edge_list  = np.argwhere(np.array(adjacency_matrix)==1)
    # edge_list = np.array([pair for pair in edge_list if pair[0] != pair[1]]) # removes self loops

    # edge_list = torch.from_numpy(edge_list)
    # edge_index = edge_list.t().contiguous()

    # return edge_index # this does not contain self-loops!
    return torch_geometric.utils.dense_to_sparse(adj)[0]

def sequence_mask(seq:torch.LongTensor, padding_idx:int=None) -> torch.BoolTensor:
    """ seq: [bsz, slen], which is padded, so padding_idx might be exist.     
    if True, '-inf' will be applied before applied scaled-dot attention"""
    return seq == padding_idx

def tensor_to_data_list(obs: torch.Tensor, adj) -> List[Data]:

    """
    Convert a tensor of shape (batch_size, ...) to a list of Data objects.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor with size (batch_size, ...).
    adj: torch.Tensor
        adjacency matrix of shape [batch_size, num_elements, num_elements] or 
        [num_elements, num_elements]. In the latter case the adjacency matrix
        is broadcasted to all instances in the batche.
        
    Returns
    -------

    out: List[Data]
        List of Data objects
    """
    batch_size = obs.shape[0]
    if adj.ndim == 2:
        adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # print(adj.shape)
    # edge_index = torch_geometric.utils.dense_to_sparse(adj)

    return Batch.from_data_list([Data(x=obs[i], edge_index= dense_to_edge_index(adj[i]) ) for i in range(obs.shape[0])])

def obs_to_geometric(obs: torch.Tensor, adj: torch.Tensor):
    """
    Transforms the observation to Data format
    class of PyTorch geometric.

    Parameters:
    -----------

    obs: torch.Tensor
        observation tensor of shape [batch_size, num_elements, feature_dim]
    adj: torch.Tensor
        adjacency matrix of shape [batch_size, num_elements, num_elements] or 
        [num_elements, num_elements]. In the latter case the adjacency matrix
        is broadcasted to all instances in the batche.

    Returns:
    --------
    """

    batch_size = obs.shape[0]
    if adj.ndim == 2:
        adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
        print(adj.shape)

    # batch = 
    
    edge_index = torch_geometric.utils.dense_to_sparse(adj)
    data = Data(x=obs, edge_index=edge_index)

    return data
    


def get_elem_action_topo_map(rllib_env: List[grid2op.Action.TopologyAction]) \
         -> Tuple[Dict[int, List[Tuple[int,int]]], Dict[int, List[Tuple[int,int]]] ]:

    """
    Creates 2 dictionaries that are usefuel for going between actions, element ids and topologies.
    1st mapping is between:
    - the element id and
    - Tuple[idx of the action, bus bar set by this action on this element]

    2ndst mapping is between:
    - the action index and 
    - Tuple[element_id, bus bar set by this action on this element]

    This is used for masking in the node level approach.
    Note 1: the input action_lst is assumed to be of the same type as the actions in the environment.
    Note 2: the action_lst can be accessed by `rllib_env.all_actions_dict`

    Parameters:
    -----------
    action_lst: List[Grid2Op action]

    Returns:
    --------
    element_to_action_num: Dict[int, List[Tuple[int,int]]]
    action_to_topology: Dict[int, List[Tuple[int,int]]]

    """
    element_to_action_num = defaultdict(list)
    action_to_topology = defaultdict(list) # action_num -> List[Tuple(element_id, bus_bar)]
    # print("Len of all actions", len(rllib_env.all_actions_dict))
    # print("Action 29", rllib_env.all_actions_dict[29])
    # for sub, sub_actions in rllib_env.env_gym.agent.sub_id_to_action.items():
    for act_num,act in enumerate(rllib_env.all_actions_dict):

        topology_changes = act.impact_on_objects()\
                ["topology"]["assigned_bus"] # list of dicts
        sub_modified = topology_changes[0]["substation"] # assumes single substation is modfied at a time
        
        modified_buses = [topo_change["bus"] for topo_change in topology_changes]
        element_ids = np.argwhere(rllib_env.org_env.grid_objects_types[:,0] == sub_modified) # going from element id per type to element id (0-55 inclusive)
        
        for element_id, bus in zip(element_ids, modified_buses):
            element_to_action_num[element_id[0]].append((act_num, bus)) # note that each element for a given substation will have the same actions
            action_to_topology[act_num].append((element_id[0], bus))
    # print("action_to_topology 29", action_to_topology[29])
    # print("Action 29", rllib_env.all_actions_dict["kurwa"])
    return element_to_action_num, action_to_topology


