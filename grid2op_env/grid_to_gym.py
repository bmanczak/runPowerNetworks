import numpy as np
import grid2op
import torch
import logging
import gym 

from grid2op.PlotGrid import PlotMatplot 
from grid2op.gym_compat import GymEnv, MultiToTupleConverter, DiscreteActSpace
from grid2op.Reward import L2RPNReward
from grid2op.Converter import IdToAct

from grid2op_env.medha_action_space import create_action_space
from grid2op_env.utils import CustomDiscreteActions

def create_gym_env(env_name = "rte_case14_realistic" , keep_obseravations = None, keep_actions = None, 
                    scale = False, convert_to_tuple = True, act_on_single_substation  = True,
                    medha_actions = True, seed=None, **kwargs):
    """
    Create a gym environment from a grid2op environment.

    Keyword arguments 1:
    ----------
    env_name: str
        Name of the grid2op environment to create.
    keep_observations: list(str)
        List of observation to keep. If None, all observation are kept.
    keep_actions: list(str)
        List of action to keep. If None, all action are kept.
        Ignored if medha_actions is True.
    scale: bool
        Not implemented yet.
    convert_to_tuple: bool
        If True, the MultiBinary actions are converted to tuple.
        Ensures compatibility with rlib. Ignored if medha_actions or 
        act_on_single_substation is True.
    medha_actions: bool
        Whether to use a custom action space as specifed by Medha.
        If True, act_on_single_substation, keep_actions and act_on_single_substation are ignored. 
    seed: int
        Seed used to initialize the environment.
    **kwargs:
        All the parameters of the grid2op environment.
        Most imporant parameters are:
        - reward_class: int

    Returns
    -------
    env_gym: GymEnv
        The gym environment.
    """

    env = grid2op.make(env_name, reward_class = L2RPNReward, test = False, **kwargs)
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)

    # Convert to gym
    env_gym = GymEnv(env)
    logging.info("Environment successfully converted to Gym")

    if keep_obseravations is not None:
        env_gym.observation_space = env_gym.observation_space.keep_only_attr(keep_obseravations)
    
    if scale: # TO-DO
        pass

    if (act_on_single_substation) and (not medha_actions):

        if keep_actions is not None:
            env_gym.action_space =DiscreteActSpace(env.action_space,
                                    attr_to_keep  = keep_actions )
        else:
            logging.info("All actions from the {env_name} environment are kept.")
            env_gym.action_space = DiscreteActSpace(env.action_space)
    
    elif not act_on_single_substation: # actions allowed on multiple substations
        logging.warning("Actions on multiple substations are allowed!")

        if keep_actions is not None:
            env_gym.action_space = env_gym.action_space.keep_only_attr(keep_actions)
    
        if convert_to_tuple: # DiscreteActSpace does not require conversion
            for action_type in env_gym.action_space:
                if type(env_gym.action_space[action_type]) == gym.spaces.multi_binary.MultiBinary:
                    env_gym.action_space = env_gym.action_space.reencode_space(action_type, MultiToTupleConverter())
                    logging.info(f"Converted action {action_type} to tuple")
        
    if medha_actions:
        logging.info("Using the action space defined by Medha")
        if env_name != "rte_case14_realistic":
            raise NotImplementedError("Medha action space is only implemented for rte_case14_realistic")

        all_actions, _ = create_action_space(env)  
        converter = IdToAct(env.action_space)  # initialize with regular the environment of the regular action space
        converter.init_converter(all_actions=all_actions) 

        env_gym.action_space = CustomDiscreteActions(converter = converter)


    return env_gym

if __name__ == "__main__":
    logging.basicConfig(filename='env_create.log', filemode='w', level=logging.INFO)

    env = create_gym_env("rte_case14_realistic", keep_obseravations= ["rho", "gen_p"], convert_to_tuple=True)