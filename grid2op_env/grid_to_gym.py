import numpy as np
import grid2op
import torch
import logging
import gym 

from grid2op.PlotGrid import PlotMatplot 
from grid2op.gym_compat import GymEnv, MultiToTupleConverter



def create_gym_env(env_name, keep_obseravtions = None, keep_actions = None, 
                    scale = False, convert_to_tuple = True, seed=None, **kwargs):
    """
    Create a gym environment from a grid2op environment.

    Parameters
    ----------
    env_name: str
        Name of the grid2op environment to create.
    keep_observations: list(str)
        List of observation to keep. If None, all observation are kept.
    keep_actions: list(str)
        List of action to keep. If None, all action are kept.
    scale: bool
        Not implemented yet.
    convert_to_tuple: bool
        If True, the MultiBinary actions are converted to tuple.
        Ensures compatibility with rlib.
    seed: int
        Seed used to initialize the environment.
    **kwargs:
        All the parameters of the grid2op environment.

    Returns
    -------
    env_gym: GymEnv
        The gym environment.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = grid2op.make(env_name, **kwargs)
    env.seed(seed)

    # Convert to gym
    env_gym = GymEnv(env)

    if keep_obseravtions is not None:
        env_gym.observation_space = env_gym.observation_space.keep_only_attr(keep_obseravtions)
    if keep_actions is not None:
        env_gym.action_space = env_gym.action_space.keep_only_attr(keep_actions)
    if scale: # TO-DO
        pass

    if convert_to_tuple:
        for action_type in env_gym.action_space:
            if type(env_gym.action_space[action_type]) == gym.spaces.multi_binary.MultiBinary:
                env_gym.action_space = env_gym.action_space.reencode_space(action_type, MultiToTupleConverter())
                logging.info(f"Converted action {action_type} to tuple")
    
    return env_gym

if __name__ == "__main__":
    logging.basicConfig(filename='env_create.log', filemode='w', level=logging.INFO)

    env = create_gym_env("rte_case14_realistic", convert_to_tuple=True)