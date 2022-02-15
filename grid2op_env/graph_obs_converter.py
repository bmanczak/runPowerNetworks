import numpy as np
import grid2op
import torch

from grid2op.gym_compat import GymEnv
from grid2op.PlotGrid import PlotMatplot

from gym.spaces import Box


def graph_converter(obs, env, hazard_threshold = 0.9):
  """
    This function converts the observation of a gym environment into a tensor
    with shape [batch_size x num_elements x num_features]. Meant as an input for the
    graph attention network.

  :param OrderedDict obs: gym observation
  :param Grid2Op_environment env: grid2op environment. Used to fetch the 
     ids of different objects.
  :param float hazard_threshold
  """

  num_elements = env.action_space.dim_topo # number of elements == number of nodes in the graph

  # rho is symmetric for both ends of the line
  rho = torch.zeros(num_elements)
  rho[env.action_space.line_or_pos_topo_vect] = torch.from_numpy(obs["rho"])
  rho[env.action_space.line_ex_pos_topo_vect] = torch.from_numpy(obs["rho"])

  # active power p
  p = torch.zeros(num_elements)
  p[env.action_space.gen_pos_topo_vect] = torch.from_numpy(obs["gen_p"]) # generator active production
  p[env.action_space.load_pos_topo_vect] = torch.from_numpy(obs["load_p"]) # load active consumption
  p[env.action_space.line_or_pos_topo_vect] = torch.from_numpy(obs["p_or"]) # origin active flow
  p[env.action_space.line_ex_pos_topo_vect] = torch.from_numpy(obs["p_ex"]) # Extremity active flow

  # overflow 
  over = torch.zeros(num_elements)
  over[env.action_space.line_or_pos_topo_vect] = torch.from_numpy(obs["timestep_overflow"]).float()
  over[env.action_space.line_ex_pos_topo_vect] = torch.from_numpy(obs["timestep_overflow"]).float()

  # powerline maintenance
  # maintenance = torch.zeros(num_elements)
  # maintenance[env.action_space.line_or_pos_topo_vect] = torch.from_numpy(obs["maintenance"]).float()
  # maintenance[env.action_space.line_ex_pos_topo_vect] = torch.from_numpy(obs["maintenance"]).float()

  # manual feature thresholding 
  hazard = torch.zeros(num_elements)
  hazard[env.action_space.line_or_pos_topo_vect] = (torch.from_numpy(obs["rho"]) > hazard_threshold).float()
  hazard[env.action_space.line_ex_pos_topo_vect] = (torch.from_numpy(obs["rho"]) > hazard_threshold).float()

  vectorized_obs = torch.stack([rho,p,over, hazard], dim = 1)

  return vectorized_obs