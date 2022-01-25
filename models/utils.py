import torch 

def vectorize_obs(obs, env_action_space, hazard_threshold = 0.9):
    """
    Vectorize the gym observation.

    :param OrderedDict obs: gym observation
    :param Grid2Op_environment env: grid2op environment. Used to fetch the 
        ids of different objects.
    :param float hazard_threshold
    """

    length = env_action_space.dim_topo # number of bus bars == number of nodes in the graph
    batch_size = obs["rho"].shape[0]
    # print("batch size", batch_size)
    # for all keys in obs print the value shape
    # for key in obs.keys():
    #     print(key, obs[key].shape)
    # rho is symmetric for both ends of the line [batch_dim, 56,1]
    rho = torch.zeros((batch_size, length))
    rho[:,env_action_space.line_or_pos_topo_vect] = obs["rho"]
    rho[:, env_action_space.line_ex_pos_topo_vect] = obs["rho"]

    # active power p [batch_dim, 56,1]
    p = torch.zeros((batch_size, length))
    p[:,env_action_space.gen_pos_topo_vect] = obs["gen_p"]# generator active production
    p[:,env_action_space.load_pos_topo_vect] = obs["load_p"] # load active consumption
    p[:,env_action_space.line_or_pos_topo_vect] = obs["p_or"] # origin active flow
    p[:,env_action_space.line_ex_pos_topo_vect] = obs["p_ex"] # Extremity active flow

    # overflow [batch_dim, 56,1]
    over = torch.zeros((batch_size, length))
    over[:,env_action_space.line_or_pos_topo_vect] = obs["timestep_overflow"].float()
    over[:,env_action_space.line_ex_pos_topo_vect] = obs["timestep_overflow"].float()

    # one-hot topo vector [batch_dim, 56,3]
    topo_vect_one_hot = torch.zeros((batch_size, length,3))
    topo_vect = obs["topo_vect"] # [batch_dim, 56,1]
    topo_vect[topo_vect==-1] = 0 # change disconneted from -1 to 0
    topo_vect_one_hot = torch.nn.functional.one_hot(topo_vect.to(torch.int64), num_classes=3)
    # print("topo_vect_one_hot", topo_vect_one_hot.shape)

    # powerline maintenance
    # maintenance = torch.zeros((batch_size, length))
    # maintenance[env_action_space.line_or_pos_topo_vect] = obs["maintenance"]).float()
    # maintenance[env_action_space.line_ex_pos_topo_vect] = obs["maintenance"]).float()

    # manual feature thresholding 
    hazard = torch.zeros((batch_size, length)) # [batch_dim, 56,1]
    hazard[:,env_action_space.line_or_pos_topo_vect] = (obs["rho"] > hazard_threshold).float()
    hazard[:,env_action_space.line_ex_pos_topo_vect] = (obs["rho"] > hazard_threshold).float()

    vectorized_obs = torch.stack([rho,p,over, hazard], dim = -1)
    # print("vectorized_obs", vectorized_obs.shape)
    vectorized_obs = torch.concat([vectorized_obs, topo_vect_one_hot], dim = -1)

    return vectorized_obs


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
        

