# gym specific, we simply do a copy paste of what we did in the previous cells, wrapping it in the
# MyEnv class, and train a Proximal Policy Optimisation based agent
from gc import callbacks
import os
import ray
import logging
import wandb
import argparse
import yaml
import random
import numpy as np
import torch 
import pickle

import gym
from gym.spaces import Discrete, Tuple, Dict, Box

from ray.rllib.models import ModelCatalog
from ray.rllib.agents import ppo, sac  # import the type of agents
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

from ray import tune
from ray.tune.registry import register_env
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

from dotenv import load_dotenv # security keys

from models.mlp import SimpleMlp
from models.substation_module import RllibSubsationModule
from models.hierarchical_agent import HierarchicalAgent, GreedySubModelNoWorker
from grid2op_env.grid_to_gym import Grid_Gym, Grid_Gym_Greedy, HierarchicalGridGym
from experiments.preprocess_config import preprocess_config, get_loader
from experiments.stopper import MaxNotImprovedStopper

from experiments.callback import LogDistributionsCallback

load_dotenv()
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")


logging.basicConfig(
    format='[INFO]: %(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in \
    function %(funcName)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)

LOCAL_DIR = "log_files"

if __name__ == "__main__":
    random.seed(2137)
    np.random.seed(2137)
    torch.manual_seed(2137)
    ModelCatalog.register_custom_model("fcn", SimpleMlp)
    ModelCatalog.register_custom_model("substation_module", RllibSubsationModule)
    ModelCatalog.register_custom_model("hierarchical_agent", HierarchicalAgent)

    register_env("Grid_Gym", Grid_Gym)
    register_env("Grid_Gym_Greedy", Grid_Gym_Greedy)
    register_env("HierarchicalGridGym", HierarchicalGridGym)
    ray.shutdown()
    ray.init(ignore_reinit_error=False)

    parser = argparse.ArgumentParser(description="Train an agent on the Grid2Op environment")
    parser.add_argument("--algorithm", type=str, default="ppo", help="Algorithm to use", choices=["ppo", "sac"])
    parser.add_argument("--algorithm_config_path", type=str, default="experiments/ppo/ppo_config.yaml", \
                                                         help="Path to config file for the algorithm")
    parser.add_argument("--use_tune", type=bool, default=True, help="Use Tune to train the agent")
    parser.add_argument("--project_name", type=str, default="testing_callback_grid", help="Name of the to be saved in WandB")
    parser.add_argument("--num_iters", type=int, default=1000, help="Number of iterations to train the agent for.")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number of workers to use for training.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to use for training.")
    parser.add_argument("--checkpoint_freq", type=int, default=10, help="Number of iterations between checkpoints.")
    parser.add_argument("--group" , type=str, default=None, help="Group to use for training.")
    parser.add_argument("--resume", type=bool, default=False, help="Resume training from a checkpoint. If yes, group must be specified.")
    parser.add_argument("--grace_period", type = int, default = 400, help = "Minimum number of timesteps before a trial can be early stopped.")
    parser.add_argument("--num_iters_no_improvement", type = int, default = 200, help = "Minimum number of timesteps before a trial can be early stopped.")
    parser.add_argument("--seed", type = int, default = -1, help = "Seed to use for training.")
    parser.add_argument("--with_opponent", type = bool, default= -1, help = "Whether to use an opponent or not.")

    args = parser.parse_args()

    logging.info("Training the agent with the following parameters:")

    for arg in vars(args):
        logging.info(f"{arg.upper()}: {getattr(args, arg)}")

    config = preprocess_config(yaml.load(open(args.algorithm_config_path), Loader=get_loader()))["tune_config"]
    if args.num_workers != -1: # overwrite config if necessary
        config["num_workers"] = args.num_workers
    if args.seed != -1:
        config["seed"] = args.seed
    if args.with_opponent != -1:
        config["env_config"]["with_opponent"] = True
        config["evaluation_config"]["env_config"]["with_opponent"] = True
    
    pretrained_model_config = config["model"]["custom_model_config"]\
                    .get("pretrained_model_config", None)
    print(pretrained_model_config)
    if pretrained_model_config is not None:
        # pretrained_substation_model = GreedySubModel(**pretrained_model_config)
        pretrained_substation_model = GreedySubModelNoWorker(model= pickle.load(open(pretrained_model_config["model_path"], "rb")),
                                                            dist_class=pickle.load(open(pretrained_model_config["dist_class_path"], "rb")),
                                                            env_config= pickle.load(open(pretrained_model_config["env_config_path"], "rb")),
                                                            num_to_sub=pickle.load(open(pretrained_model_config["num_to_sub_path"], "rb")))

        config["model"]["custom_model_config"]["pretrained_substation_model"] = pretrained_substation_model
        config["model"]["custom_model_config"].\
            pop("pretrained_model_config", None) # pretrained model extracted
    
    print("Config is", config)
    if args.algorithm == "ppo":
        trainer = ppo.PPOTrainer
    elif args.algorithm == "sac":
        trainer = sac.SACTrainer
    else:
        raise ValueError("Unknown algorithm. Choices are: ppo, sac")
    # print("The final env config is:")
    # print(config["env_config"])
    env_config_train = config["env_config"]
    env_config_val = config["evaluation_config"]["env_config"]

    grid_gym = Grid_Gym(env_config_train)
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith("choose_action"):
            return "choose_action_agent"
        else:
            return "choose_substation_agent"
    
    config = {
            "env": HierarchicalGridGym,
            "env_config": env_config_train,
            "multiagent": {
                "policies": {
                    "choose_substation_agent": (
                        PPOTorchPolicy,
                        grid_gym.observation_space,
                        Discrete(8),
                            {"gamma": 0.99,
                            "lr": 0.0001, #!choice [0.001, 0.0001] #tune.grid_search([1e-3 1e-41e-5])
                            "kl_coeff": 0.2, # !choice [0.15, 0.3, 0.2, 0.25]
                            "lambda":  0.95, # !quniform [0.94, 0.96, 0.01] 
                            "vf_loss_coeff": 0.9 ,#!quniform [0.75,1,0.05]
                            "vf_clip_param": 1500 ,#!choice [100, 500, 1500, 2000]
                            "rollout_fragment_length": 2000 ,#!choice [64,128,200] # 16
                            "sgd_minibatch_size": 128, #256 #!choice [256,512] # 64
                            "train_batch_size": 256, #1024 #!choice [1024, 2048] #2048
                            "num_sgd_iter": 5,
                            "entropy_coeff": 0.01,
                            "model": {
                                "fcnet_hiddens": [256,256,256],
                                "fcnet_activation": "relu",
                                "custom_model" : "fcn",
                                "custom_model_config" : {
                                    "use_parametric": False,
                                    "env_obs_name": "grid"
                                }
                            }
                        },
                    ),
                    "choose_action_agent": (
                        PPOTorchPolicy,
                        gym.spaces.Dict({
                            "action_mask":Box(0, 1, shape=(grid_gym.action_space.n, ), dtype=np.float32),
                            "regular_obs": grid_gym.observation_space,
                            # "grid": gym.spaces.Tuple([grid_gym.observation_space, Discrete(8)])
                            "chosen_substation": Discrete(8)
                        }) ,
                        grid_gym.action_space,
                        
                            {"gamma": 0.99,
                            "lr": 0.0001, #!choice [0.001, 0.0001] #tune.grid_search([1e-3 1e-41e-5])
                            "kl_coeff": 0.2, # !choice [0.15, 0.3, 0.2, 0.25]
                            "lambda":  0.95, # !quniform [0.94, 0.96, 0.01] 
                            "vf_loss_coeff": 0.9 ,#!quniform [0.75,1,0.05]
                            "vf_clip_param": 1500 ,#!choice [100, 500, 1500, 2000]
                            "rollout_fragment_length": 2000 ,#!choice [64,128,200] # 16
                            "sgd_minibatch_size": 128, #256 #!choice [256,512] # 64
                            "train_batch_size": 256, #1024 #!choice [1024, 2048] #2048
                            "num_sgd_iter": 5,
                            "entropy_coeff": 0.01,
                            "model": {
                                "fcnet_hiddens": [256,256,256],
                                "fcnet_activation": "relu",
                                "custom_model" : "fcn",
                                "custom_model_config" : {
                                    "use_parametric": True,
                                    "env_obs_name": ["regular_obs", "chosen_substation"]
                                }
                            }
                        },
                    ),
                },
                "policy_mapping_fn": policy_mapping_fn,
            },
            "num_workers": 2,
            "framework": "torch",
            #  "rollout_fragment_length": 128 ,#!choice [64,128,200] # 16
            # "sgd_minibatch_size": 256, #256 #!choice [256,512] # 64
            # "train_batch_size": 1024, #1024 #!choice [1024, 2048] #2048
            # "num_sgd_iter": 5,
            "callbacks": LogDistributionsCallback,
            "evaluation_interval": 10,
            "evaluation_num_episodes" : 100,
            "evaluation_config": { 
                "env": HierarchicalGridGym, 
                "env_config": env_config_val # use the validation env
            }
        } 

    if args.use_tune:
        # Limit the number of rows.
        reporter = CLIReporter()
        stopper = CombinedStopper(
            MaximumIterationStopper(max_iter = args.num_iters),
            MaxNotImprovedStopper(metric = "episode_reward_mean",
                                    grace_period = args.grace_period, 
                                    num_iters_no_improvement = args.num_iters_no_improvement)
                                    )
        
        analysis = ray.tune.run(
                trainer,
                progress_reporter = reporter,
                config = config,
                name = args.group,
                local_dir= LOCAL_DIR,
                checkpoint_freq=args.checkpoint_freq,
                stop = stopper,
                checkpoint_at_end=True,
                num_samples = args.num_samples,
                callbacks=[WandbLoggerCallback(
                            project=args.project_name,
                            group = args.group,
                            api_key =  WANDB_API_KEY,
                            log_config=True)],
                # loggers= [CustomTBXLogger],
                keep_checkpoints_num = 5,
                checkpoint_score_attr="evaluation/episode_reward_mean",
                verbose = 1,
                resume = args.resume
                )
        ray.shutdown()
    else: # use ray trainer directly
        trainer_object = trainer(env=Grid_Gym,
                 config=config)
        
        for step in range(args.num_iters):
            result = trainer_object.train()
            print(result["episode_len_mean"], flush = True)
            if (step+1) % args.checkpoint_freq == 0:
                checkpoint = trainer_object.save()
                print("checkpoint saved at", checkpoint)
            print("-"*40, flush = True)
    

    