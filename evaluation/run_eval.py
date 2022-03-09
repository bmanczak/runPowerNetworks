import argparse
import os
import numpy as np
from tqdm import tqdm
import logging

import models

from ray.rllib.models import ModelCatalog
from collections import defaultdict
from tqdm import tqdm
from typing import Tuple

from grid2op_env.grid_to_gym import Grid_Gym, Grid_Gym_Greedy, create_gym_env
from evaluation.restore_agent import restore_agent
from evaluation.rllib_to_grid import AgentThresholdEnv, AgentThresholdEnvGreedy, ClassicGreedyWrapper
from models.mlp import SimpleMlp
from models.greedy_agent import ClassicGreedyAgent


np.random.seed(0)
ModelCatalog.register_custom_model("fcn", SimpleMlp)

class EvaluationRunner:
    """
    This class unifies the evaluation of the greedy, ppo or sac agents.
    It saves the action, topological vectors and number of steps completed.
    """

    def __init__(self, agent_type:str, checkpoint_path:str, checkpoint_num = None,
             nb_episode:int = 1000, save_path:str = None, random_sample:bool = False, 
             pbar:bool = False, greedy_env_config_agent_type:str = None):

        """

        Parameters:
        ----------
        agent_type: str
            The type of the agent. Supported types are:
            - "ppo": The agent is a PPO agent.
            - "sac": The agent is a SAC agent
            - "greedy": The agent is a greedy agent. Note that the checkpoint path
                to a PPO or a SAC model must be provided to fetch the env_config.
        checkpoint_path: str
            The path to the checkpoint.
        checkpoint_num: int
            The number of the checkpoint.
        nb_episode: int
            The number of episode to evaluate
        save_path: str 
            The path to save the results.
        random_sample: bool
            If True the agent will be trained with a random sample of the chronics.
        pbar: bool
            If True a progress bar will be displayed. Not recommended with Lisa.
        greedy_env_config_agent_type: str
            The type of the agenet from which the environment is fetched.
        """

        self.agent_type = agent_type
        self.checkpoint_path = checkpoint_path
        self.checkpoint_num = checkpoint_num
        self.nb_episode = nb_episode
        self.save_path = save_path
        self.random_sample = random_sample
        self.pbar = pbar

        assert self.agent_type in ["ppo", "sac", "greedy"], "Agent type not supported"
        if agent_type =="greedy":
            assert greedy_env_config_agent_type in ["ppo", "sac"], "Specify type of agent for greedy environment"
            self.greedy_env_config_agent_type = greedy_env_config_agent_type

        if self.random_sample and nb_episode < 1000:
            self.chronics_to_study = np.random.randint(1, 1001, nb_episode)
        else:
            self.chronics_to_study = range(1, nb_episode + 1)
        
        print(f"Evaluating the following chronics {list(self.chronics_to_study)}")
        
        self.chronics_to_study = tqdm(self.chronics_to_study) if self.pbar else self.chronics_to_study

        # Execute the steps needed for evaluation loop
        self.restore_agent()
        self.load_env_wrapper()
        self.wrap_agent()

        if save_path is None:
            if self.agent_type == "greedy":
                self.save_path = os.path.join("evaluation/eval_results", f'greedy_{checkpoint_path.split("/")[-1]}')
            else:
                self.save_path = os.path.join("evaluation/eval_results", f'{checkpoint_path.split("/")[-1]}')
            self.save_path = f"{self.save_path}_{self.checkpoint_num}_{self.nb_episode}_{self.random_sample}"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def restore_agent(self):
        """
        Restores the PPO or SAC agent from a checkpoint.
        """
        agent_type = self.agent_type if self.agent_type != "greedy" else self.greedy_env_config_agent_type
        agent, env_config = restore_agent(path = self.checkpoint_path,
                   checkpoint_num = self.checkpoint_num,
                   trainer_type= agent_type)  
    
        self.agent, self.env_config  = agent, env_config

    def load_env_wrapper(self):
        """
        Loads the envrionment wrapper.
        """
        env_config = self.env_config

        if env_config.get("greedy_agent", False):
            self.rllib_env = Grid_Gym_Greedy(env_config)
        else:
            self.rllib_env = Grid_Gym(env_config)
        
        # Load the grid2op environment from wrapper
        self.env = self.rllib_env.org_env
    
    def wrap_agent(self):
        """
        Wraps the agent fro evaluation with the Grid2Op environment.
        """
       
        if self.agent_type in ["ppo", "sac"]:
            if self.env_config.get("greedy_agent", False): # substation + greedy
                self.wrapped_agent = AgentThresholdEnvGreedy(self.rllib_env, self.agent,
                                rho_threshold = self.env_config["rho_threshold"])
            else:
                self.wrapped_agent = AgentThresholdEnv(self.rllib_env, self.agent,
                                rho_threshold = self.env_config["rho_threshold"])
        else: # greedy
            print("Using totally greedy agent")
            greedy_agent = self.instanciate_greedy(self.env_config)
            self.wrapped_agent = ClassicGreedyWrapper(greedy_agent, self.env_config["rho_threshold"])
    
    def eval_loop(self) -> Tuple[dict, dict, dict]:

        actions = defaultdict(list)
        topo_vects = defaultdict(list) 
        chronic_to_num_steps = defaultdict(list) 
        for chronic_id in self.chronics_to_study:
            self.env.set_id(chronic_id)

            if int(self.env.chronics_handler.get_name()) > len(self.env.chronics_handler.subpaths): 
                raise ValueError("Chronics id is too high")
            assert int(self.env.chronics_handler.get_name()) == chronic_id-1, "Chronics id is not the same as the one in the environment"
            
            done = False
            num_steps = 0
            obs = self.env.reset()
            add_obs = False
            
            while not done:
                num_steps += 1
                act = self.wrapped_agent.act(obs)
                if obs.rho.max() > self.env_config["rho_threshold"]: # save action taken above the threshold
                    add_obs = True
                    actions[chronic_id].append(act)
                    
                obs, reward, done, info = self.env.step(act)

                if add_obs: # save topo vector resulting from action above the threshold
                    topo_vects[chronic_id].append(obs.topo_vect)
                    add_obs = False
                    chronic_to_num_steps[chronic_id].append(num_steps)

            chronic_to_num_steps[chronic_id].append(num_steps)
        
        np.save(os.path.join(self.save_path, f"actions.npy"),
                actions)
        np.save(os.path.join(self.save_path, f"topo_vects{self.checkpoint_num}_{self.nb_episode}_{self.random_sample}.npy"),
                topo_vects)
        np.save(os.path.join(self.save_path, f"chronic_to_num_steps{self.checkpoint_num}_{self.nb_episode}_{self.random_sample}.npy"),
                chronic_to_num_steps)
        
        print("Mean number of steps completed over the tested chronis", np.mean([steps[-1] for chronic,steps in chronic_to_num_steps.items()]))  
        
        return actions, topo_vects, chronic_to_num_steps    

    @staticmethod
    def instanciate_greedy(env_config: dict) -> models.greedy_agent.ClassicGreedyAgent :
        """
        Instanciates the greedy agent from env_config
        """
        _, do_nothing_actions, org_env, all_actions_dict = create_gym_env(env_name = env_config["env_name"],
                                        keep_obseravations= env_config["keep_observations"],
                                        keep_actions= env_config["keep_actions"],
                                        convert_to_tuple=env_config["convert_to_tuple"],
                                        act_on_single_substation=env_config["act_on_single_substation"],
                                        medha_actions=env_config["medha_actions"],
                                        scale = env_config.get("scale", False),
                                        disable_line = env_config.get("disable_line", -1),
                                        conn_matrix = env_config.get("conn_matrix", False),
                                        substation_actions = False,
                                        greedy_agent = False, # has to be false to return the do nothing actions properly
                                        graph_obs = env_config.get("graph_obs", False),
                                        combine_rewards= env_config.get("combine_rewards", False))
               
        greedy_agent = ClassicGreedyAgent(org_env.action_space, action_list=all_actions_dict, do_nothing_idx=do_nothing_actions)

        return greedy_agent



if "__main__" == __name__:
    ModelCatalog.register_custom_model("fcn", SimpleMlp)

    parser = argparse.ArgumentParser(description="Run a grid2op evaluation")
    parser.add_argument("--agent_type", type=str, default="ppo", help="What type of model is being evaluated")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint to restore")
    parser.add_argument("--checkpoint_num", type=int, default=None, help="Number of the checkpoint to restore")
    parser.add_argument("--nb_episode", type=int, default=1000, help="Number of episode to run")
    parser.add_argument("--save_path", type=str, default=None, help="Path to the folder where to save the results")
    parser.add_argument("--random_sample", type=bool, default=False, help="Random sample episode id")
    parser.add_argument("--pbar", type=bool, default=True, help="Use a progress bar")
    parser.add_argument("--greedy_env_config_agent_type", type=str, default="ppo", help="The type of the agenet from which the environment is fetched.")
    
    args = parser.parse_args()

    eval_runner = EvaluationRunner(**vars(args))
    eval_runner.eval_loop()
    
