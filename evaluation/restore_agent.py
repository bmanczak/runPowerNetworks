import logging
import re 
import json
import os
import logging
import numpy as np
import gym

from train_hierarchical import policy_mapping_fn

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo, sac # import the type of agents
from grid2op_env.grid_to_gym import Grid_Gym, HierarchicalGridGym
from typing import Tuple, Union, Optional


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def restore_agent(trainer_type:str, path:str, checkpoint_num:Optional[int] = None,
                 modify_keys: Optional[dict] = None, hierachical = False) \
                        -> Tuple[Union[ppo.PPOTrainer, sac.SACTrainer], dict]:
    """
    Function that restores the agent from tune checkpointwith 
    correct hyperparameters.
    
    Keyword arguments:
    ----------
    trainer_type: str 
        Type of trainer to use. "ppo" and "sac" are supported.
    path: str
        Path that contains (1) the params.json and 
        (2) folders that contain checkpoints. Note that
        default tune naming scheme is assumed for (2), i.e.
        checkpoint_<0..0checkpoint_num> that contains files
        checkpoint-<checkpoint_num> and checkpoint-<checkpoint_num>.tune_metadata  
    checkpoint_num: int (Optional)
        What checkpoint number to restore. If None,
        the checkpoint with the highest number is restored.
    modify_keys: dict (Optional)
        Keys to add/change in the config.
    hierarchical: bool (Optional)
        If True, the agent is restored with the hierarchical spec.
    
    Returns:
    --------
    agent: rllib agent
        Agent restored from the checkpoint.
    env_config: dict
        Env config if return_env_config is True.


    Example usage:
    ----------
    agent, env_config = restore_agent(path = "/Users/blazejmanczak/Desktop/try_this/PPO_Grid_Gym_e07fb_00004_4_clip_param=0.2,lambda=0.94,lr=0.001,vf_loss_coeff=0.9_2021-12-04_03-59-36",
                   checkpoint_num = 900,
                   modify_keys={"env_config": {"scale": True}})

    rllib_env = Grid_Gym(env_config);
    """

    config_params = json.load(open(os.path.join(path, "params.json")))
    config_params.pop("callbacks", None)
    

    if hierachical:
        grid_gym = Grid_Gym(config_params["env_config"])
        config_params["multiagent"]["policies"]["choose_substation_agent"] = (
        PPOTorchPolicy,
        # grid_gym.observation_space,
        gym.spaces.Dict({
            "regular_obs": grid_gym.observation_space,
            "chosen_action": Discrete(grid_gym.action_space.n),
        }) ,
        Discrete(8),
        config_params["multiagent"]["policies"]["choose_substation_agent"][3]
        
    )

    config_params["multiagent"]["policies"]["choose_action_agent"] = (
        PPOTorchPolicy,
        gym.spaces.Dict({
            "action_mask":Box(0, 1, shape=(grid_gym.action_space.n, ), dtype=np.float32),
            "regular_obs": grid_gym.observation_space,
            "chosen_substation": Discrete(8)
        }) ,
        grid_gym.action_space,
        config_params["multiagent"]["policies"]["choose_action_agent"][3]
    )

    config_params["multiagent"]["policy_mapping_fn"] = policy_mapping_fn


    # Optionally modify the keys 
    if modify_keys is not None:
        for key, val in modify_keys.items():
            if type(val) == dict:
                for key2, val2 in val.items():
                    logging.warning(f"Changing config for key {key}[{key2}] from {config_params[key][key2]} to {val2}")
                    config_params[key][key2] = val2
                    print(f"Changing config {key}[{key2}] to {val2}")
            else:
                logging.warning(f"Changing config for key {key} from {config_params.get(key, None)} to {val}")
                config_params[key] = val
                
    env_config = config_params["env_config"]

    # Get the checkpoint path with the highest number
    checkpoint_paths= os.listdir(path)
    if checkpoint_num is None: # get the highest checkpoint number
        checkpoint_num = max([int(re.search(r"checkpoint_0*(\d+)", checkpoint_paths[i]).group(1)) \
                         for i in range(len(checkpoint_paths)) if checkpoint_paths[i][0:10]=="checkpoint"])

    checkpoint_path = os.path.join(path,f"checkpoint_{str(0)*(6-len(str(checkpoint_num)))}{checkpoint_num}", \
                      f"checkpoint-{checkpoint_num}")

    logger.info(f"Restoring checkpoint {checkpoint_num} from {checkpoint_path}")

    if trainer_type == "ppo":
        if hierachical:
            agent = ppo.PPOTrainer(env = HierarchicalGridGym,
                        config=config_params)
        else:
            agent = ppo.PPOTrainer(env=Grid_Gym,
                config = config_params)
    elif trainer_type == "sac":
        agent = sac.SACTrainer(env=Grid_Gym,
            config = config_params)
            
    agent.restore(checkpoint_path);

    return agent, env_config
