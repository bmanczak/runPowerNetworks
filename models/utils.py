import torch 

def vectorize_obs(obs, env, hazard_threshold = 0.9):
   """
   Vectorize the gym observation.

   :param OrderedDict obs: gym observation
   :param Grid2Op_environment env: grid2op environment. Used to fetch the 
      ids of different objects.
   :param float hazard_threshold
   """

   length = env.action_space.dim_topo # number of bus bars == number of nodes in the graph

   # rho is symmetric for both ends of the line [56,1]
   rho = torch.zeros(length)
   rho[env.action_space.line_or_pos_topo_vect] = torch.from_numpy(obs["rho"])
   rho[env.action_space.line_ex_pos_topo_vect] = torch.from_numpy(obs["rho"])

   # active power p [56,1]
   p = torch.zeros(length)
   p[env.action_space.gen_pos_topo_vect] = torch.from_numpy(obs["gen_p"]) # generator active production
   p[env.action_space.load_pos_topo_vect] = torch.from_numpy(obs["load_p"]) # load active consumption
   p[env.action_space.line_or_pos_topo_vect] = torch.from_numpy(obs["p_or"]) # origin active flow
   p[env.action_space.line_ex_pos_topo_vect] = torch.from_numpy(obs["p_ex"]) # Extremity active flow

   # overflow [56,1]
   over = torch.zeros(length)
   over[env.action_space.line_or_pos_topo_vect] = torch.from_numpy(obs["timestep_overflow"]).float()
   over[env.action_space.line_ex_pos_topo_vect] = torch.from_numpy(obs["timestep_overflow"]).float()

   # one-hot topo vector [56,3]
   topo_vect_one_hot = torch.zeros(length,3)
   topo_vect = obs["topo_vect"]
   topo_vect[topo_vect==-1] = 0 # change disconneted from -1 to 0
   topo_vect_one_hot = torch.nn.functional.one_hot(torch.from_numpy(topo_vect).to(torch.int64), num_classes=3)

   # powerline maintenance
   # maintenance = torch.zeros(length)
   # maintenance[env.action_space.line_or_pos_topo_vect] = torch.from_numpy(obs["maintenance"]).float()
   # maintenance[env.action_space.line_ex_pos_topo_vect] = torch.from_numpy(obs["maintenance"]).float()

   # manual feature thresholding 
   hazard = torch.zeros(length) # [56,1]
   hazard[env.action_space.line_or_pos_topo_vect] = (torch.from_numpy(obs["rho"]) > hazard_threshold).float()
   hazard[env.action_space.line_ex_pos_topo_vect] = (torch.from_numpy(obs["rho"]) > hazard_threshold).float()

   vectorized_obs = torch.stack([rho,p,over, hazard], dim = 1)
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
        

