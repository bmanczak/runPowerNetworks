import logging
import re 
import json
import os

from ray.rllib.agents import ppo, sac # import the type of agents
from grid2op_env.grid_to_gym import Grid_Gym

def restore_agent(path, checkpoint_num = None, modify_keys = None, return_env_config = True, trainer_type = "ppo"):
    """
    Function that restores the agent from tune checkpointwith 
    correct hyperparameters.
    
    Keyword arguments:
    ----------
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
    return_env_config: bool (Optional)
        If True, returns the env_config.
    trainer_type: str (Optional)
        Type of trainer to use. "ppo" and "sac" are supported.
    
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
                   modify_keys={"env_config": {"scale": True}},
                   return_env_config = True)

    rllib_env = Grid_Gym(env_config);
    """

    config_params = json.load(open(os.path.join(path, "params.json")))
    config_params.pop("callbacks", None)
    env_config = config_params["env_config"]
    # Optionally modify the keys 
    if modify_keys is not None:
        for key, val in modify_keys.items():
            if type(val) == dict:
                for key2, val2 in val.items():
                    config_params[key][key2] = val2
                    logging.info(f"Setting config {key}[{key2}] to {val2}")
                    print(f"Setting config {key}[{key2}] to {val2}")
            else:
                config_params[key] = val
                logging.info(f"Setting config {key} to {val}")

    # Get the checkpoint path with the highest number
    checkpoint_paths= os.listdir(path)
    if checkpoint_num is None: # get the highest checkpoint number
        checkpoint_num = max([int(re.search(r"checkpoint_0*(\d+)", checkpoint_paths[i]).group(1)) \
                         for i in range(len(checkpoint_paths)) if checkpoint_paths[i][0:10]=="checkpoint"])

    checkpoint_path = os.path.join(path,f"checkpoint_{str(0)*(6-len(str(checkpoint_num)))}{checkpoint_num}", \
                      f"checkpoint-{checkpoint_num}")
    print(f"Restoring checkpoint {checkpoint_num} from {checkpoint_path}")

    if trainer_type == "ppo":
        agent = ppo.PPOTrainer(env=Grid_Gym,
            config = config_params)
    elif trainer_type == "sac":
        agent = sac.SACTrainer(env=Grid_Gym,
            config = config_params)

    agent.restore(checkpoint_path);

    if return_env_config:
        return agent, env_config
    else:
        return agent
    
