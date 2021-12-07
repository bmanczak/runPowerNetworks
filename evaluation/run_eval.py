import argparse
import os
import numpy as np
from re import S

from grid2op.Agent import DoNothingAgent
from grid2op.Runner import Runner
from ray.rllib.models import ModelCatalog

from grid2op_env.grid_to_gym import Grid_Gym
from evaluation.restore_agent import restore_agent
from rllib_to_grid import AgentFromGym
from models.mlp import SimpleMlp




def run_eval(agent_type = "ppo", checkpoint_path = None, checkpoint_num = None,
             nb_episode = 10, save_path = None, nb_core = 1,
            episode_id = False, scale_obs = True, random_sample = False):

    """
    This function will run the evaluation of the agent.

    Parameters:
    ----------
    agent_type: str
        The type of the agent. Supported types are:
        - "ppo": The agent is a PPO agent.
        - "dn": The agent is a DoNothing agent.
    checkpoint_path: str
        The path to the checkpoint.
    checkpoint_num: int
        The number of the checkpoint.
    nb_episode: int
        The number of episode to evaluate
    save_path: str 
        The path to save the results.
    nb_core: int
        The number of core to use.
    episode_id: list
        The list of episodes to evaluate.
        If not None must be equal to nb_episode.
    scale_obs: bool
        If True the observation will be scaled with 
        the scaling arrays from grid2op_env
    
    Raises:
        ValueError: If the agent_type is not supported.
        ValueError: checkpoint_path is None and agent type is not "dn".
    """
    np.random.seed(42)

    if random_sample:
        # make nb_episdode a random sample array of integers of size nb_episode id from 0 to 999
        episode_id = np.random.randint(0, 1000, nb_episode)

    if checkpoint_path is None and agent_type != "dn":
        raise ValueError("You must specify the path to the checkpoint.")

    
    if save_path is None:
        if agent_type == "dn":
            save_path = os.path.join("runner_log", "dn")
        else:
             save_path = os.path.join("runner_log", checkpoint_path.split("/")[-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    agent, env_config = restore_agent(path = checkpoint_path,
                   checkpoint_num = checkpoint_num,
                   modify_keys={"env_config": {"scale": scale_obs}},
                   return_env_config = True)

    rllib_env = Grid_Gym(env_config)
    wrapped_agent = AgentFromGym(rllib_env, agent)

    if agent_type =="ppo":
        runner = Runner(**rllib_env.org_env.get_params_for_runner(), agentClass=None, agentInstance=wrapped_agent)
    
    elif agent_type == "dn":
        runner = Runner(**rllib_env.org_env.get_params_for_runner(), agentClass=DoNothingAgent, agentInstance=None)
    
    else:
        raise ValueError("Agent type not recognized. Avaliable options are 'ppo' and 'dn'")

    res = runner.run(nb_episode=nb_episode, 
                 nb_process=nb_core,
                 path_save=save_path, 
                env_seeds = [42]*nb_episode,
                episode_id= episode_id,
                pbar = True)


    for _, chron_id, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics with id {}\n".format(chron_id)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)


if "__main__" == __name__:
    ModelCatalog.register_custom_model("fcn", SimpleMlp)

    parser = argparse.ArgumentParser(description="Run a grid2op evaluation")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint to restore")
    parser.add_argument("--checkpoint_num", type=int, default=None, help="Number of the checkpoint to restore")
    parser.add_argument("--nb_episode", type=int, default=10, help="Number of episode to run")
    parser.add_argument("--save_path", type=str, default=None, help="Path to the folder where to save the results")
    parser.add_argument("--nb_core", type=int, default=1, help="Number of core to use")
    parser.add_argument("--episode_id", type=list, default=None, help="Id of the episode to run")
    parser.add_argument("--scale_obs", type=bool, default=True, help="Scale the observation")
    parser.add_argument("--random_sample", type=bool, default=False, help="Random sample episode id")
    args = parser.parse_args()

    run_eval(**vars(args))
