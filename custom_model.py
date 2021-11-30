# gym specific, we simply do a copy paste of what we did in the previous cells, wrapping it in the
# MyEnv class, and train a Proximal Policy Optimisation based agent
from typing import OrderedDict
import gym
import os
import ray
import gym
import numpy as np
import grid2op 

import torch 
import torch.nn as nn
import logging
#import wandb

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.agents import ppo  # import the type of agents

from grid2op_env.grid_to_gym import create_gym_env
from grid2op.gym_compat import GymEnv, MultiToTupleConverter
from grid2op.Reward import L2RPNReward

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

from ray import tune
from ray.tune.registry import register_env
from ray.tune.integration.wandb import WandbLoggerCallback
from dotenv import load_dotenv

load_dotenv()
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

from custom_trainer import CustomPPOTrainer
from custom_policy import CustomPPOTorchPolicy
# nice article about Ray https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5

#wandb.init(project="grid2op_rlib", entity="bmanczak")

# Limit values suitable for use as close to a -inf logit. These are useful
# since -inf / inf cause NaNs during backprop.
FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

class Grid_Gym(gym.Env):
    def __init__(self, env_config):
        
        self.env_gym, self.do_nothing_actions = create_gym_env(env_name = env_config["env_name"],
                                        keep_obseravations= env_config["keep_observations"],
                                        keep_actions= env_config["keep_actions"],
                                        convert_to_tuple=env_config["convert_to_tuple"],
                                        act_on_single_substation=env_config["act_on_single_substation"],
                                        medha_actions=env_config["medha_actions"])
        
        # Define parameters needed for parametric action space
        self.rho_threshold = env_config.get("rho_threshold", 0.95) - 1e-5 # used for stability in edge cases
        self.parametric_action_space = env_config["use_parametric"] and env_config["medha_actions"] and "rho" in env_config["keep_observations"]
        logging.info(f"Using parametric action space:", self.parametric_action_space)
        print(f"Using parametric action space:", self.parametric_action_space)

        # Transform the gym action and obsrvation space that is c
        if env_config["act_on_single_substation"] or env_config["medha_actions"]:
            self.action_space = gym.spaces.Discrete(self.env_gym.action_space.n) # then already discrete
        else:
            # First serialise as dict, then convert to Dict gym space
            # for the sake of compatibility with Ray Tune
            a = {k: v for k, v in self.env_gym.action_space.items()}
            self.action_space = gym.spaces.Dict(a)
        
        # Transform the observation space
        d = {k: v for k, v in self.env_gym.observation_space.items()}
        if self.parametric_action_space: # the actions must be discrete for this
            self.observation_space = gym.spaces.Dict({
                                        "action_mask" :gym.spaces.Box(0, 1, shape=(self.env_gym.action_space.n, ), dtype=np.float32),
                                        "grid" : gym.spaces.Dict(d)})
                                       
        else:
            self.observation_space = gym.spaces.Dict(d)

        print("CREATED observation space", self.observation_space)
        #print("CREATED observation space", self.observation_space["action_mask"].shape)



    def update_avaliable_actions(self, mask_topo_change):
        """
        Masks the actions that change the topology of the grid.

        Args:
            mask_topo_change (bool): if True, only the do-nothing actions are not masked.
        """
        if mask_topo_change:
            self.action_mask = np.array(
                                            [0.] * self.env_gym.action_space.n, dtype=np.float32)
            self.action_mask[0] = 1. # hack: the 0-th action is doing nothing for all configurations.
        else:
            self.action_mask = np.array(
                                        [1.] * self.env_gym.action_space.n, dtype=np.float32)
       # print("THE ACTIONS MAKS SHAPE", self.action_mask.shape)

        
    def reset(self):
        """
        Reset the environment such that the first observation after reset has some line 
        above the pre-specifed threshold.
        """

        obs = self.env_gym.reset()
        if self.parametric_action_space:
            mask_topo_change = max(obs["rho"]) > self.rho_threshold
            self.update_avaliable_actions(mask_topo_change)
            # print(f'Required shape: {self.observation_space["action_mask"].shape}, state: {self.action_mask.shape}')
            # print(f'low: {self.observation_space["action_mask"].low}, checking: {np.all(self.action_mask >= self.observation_space["action_mask"].low)}')
            # print(f'high: {self.observation_space["action_mask"].high}, checking: {np.all(self.action_mask <= self.observation_space["action_mask"].high)}')
            #print(type(obs["gen_p"][0]))
            return {"action_mask": self.action_mask, "grid": obs}
       
        return obs

        # observed_max_rho = max(obs["rho"]) # check for the highest relative load
        
        # while (observed_max_rho < self.rho_threshold): # if the highest relative load is still under threshold
        #     obs, _, done, _ = self.env_gym.step(self.env_gym.action_space.sample()) # random actions get the agent to the bad overloaded state quicker
        #     observed_max_rho = max(obs["rho"])
        #     if done: # query again beacuse the current episode is over
        #         obs = self.env_gym.reset()
        #         observed_max_rho = max(obs["rho"])
        
        print("obs", obs)
        print("-"*40)
        return obs

    def step(self, action):
        #cum_reward = 0 # amass the reward for the action and the do nothing actions
        obs, reward, done, info = self.env_gym.step(action) 
        if self.parametric_action_space:
            mask_topo_change = max(obs["rho"]) > self.rho_threshold
            self.update_avaliable_actions(mask_topo_change)
            obs = {"action_mask": self.action_mask, "grid": obs}


        return obs, reward, done, info

        # cum_reward += reward 
        # observed_max_rho = max(obs["rho"])

        # if not done: # if agent remedied the situtaion then do no nothing unitl high load appears 
        #     while (observed_max_rho < self.rho_threshold): 
        #         obs, reward, done, info = self.env_gym.step(self.do_nothing_actions[0]) # all do nothing actions do not do anything
        #         cum_reward += reward
        #         observed_max_rho = max(obs["rho"])
        #         if done: # end of episode
        #             break
                    
        return obs, reward, done, info



class SimpleMlp(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        """
        Initialize the model.

        Parameters:
        ----------
        obs_space: gym.spaces.Space
            The observation space of the environment.
        action_space: gym.spaces.Space
            The action space of the environment.
        num_outputs: int
            The number of outputs of the model.

        model_config: ModelConfigDict
            The configuration of the model as passed to the rlib trainer.
        name: str
            The name of the model captured in model_config["model_name"]
        """
        
        # Call the parent constructor.
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        
        # Fetch the network specification
        hiddens = list(model_config.get("fcnet_hiddens", [])) + \
            list(model_config.get("post_fcnet_hiddens", []))

        # activation = model_config.get("fcnet_activation")
        # no_final_linear = model_config.get("no_final_linear") # if True, last int fcnet_hiddens should match num_outputs
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.parametric_action_space = model_config["custom_model_config"].get("use_parametric", False)
        self.env_obs_name = model_config["custom_model_config"].get("env_obs_name", "grid")
      
        logging.info(f"Using parametric action space:", self.parametric_action_space)
       
        layers = []
        # print("THIS IS the observation space", obs_space)
        # print("this is the action space number of aciton", action_space.n)
        if self.parametric_action_space: # do not parametrize the action mask
            prev_layer_size = int(np.product(obs_space.shape) - action_space.n) # dim of the observation space
        else:
            prev_layer_size = int(np.product(obs_space.shape)) # dim of the observation space
        

        # Create hidden layers
        for size in hiddens:
            #print("prev_layer_size", prev_layer_size)
            layers += [nn.Linear(prev_layer_size, size), nn.ReLU(inplace=True)]
            prev_layer_size = size
        
        
        self._logits = nn.Linear(prev_layer_size, num_outputs)
        self._hidden_layers = nn.Sequential(*layers)
        
        # Value function spec
        self._value_branch_separate = None
        if not self.vf_share_layers: # if we want to separate value function
            # Build a parallel set of hidden layers for the value net.
            if self.parametric_action_space: # do not parametrize the action mask
                prev_vf_layer_size = int(np.product(obs_space.shape) - action_space.n) # dim of the observation space
            else:
                prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers += [nn.Linear(prev_vf_layer_size, size), nn.ReLU(inplace=True)]
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = nn.Linear(prev_layer_size, 1)
  
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType):

        if self.parametric_action_space:
            obs = torch.concat(
                            [val for val in input_dict["obs"][self.env_obs_name].values()], dim=1) # [BATCH_DIM, obs_dim]
            #print("obs", obs.shape, obs)
            inf_mask = torch.clamp(torch.log(input_dict["obs"]["action_mask"]), FLOAT_MIN, FLOAT_MAX)
        else:
            obs = input_dict["obs_flat"].float()

        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)

        logits = self._logits(self._features) 
        if self.parametric_action_space:
            logits += inf_mask
        #print("logits shape", logits.shape)
        return logits, state

    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)


if __name__ == "__main__":
    
    ModelCatalog.register_custom_model("fcn", SimpleMlp)
    register_env("Grid_Gym", Grid_Gym)

    ray.init(ignore_reinit_error=True)

    # Configure the model

    model_config = {
            "fcnet_hiddens": [128,128, 128],
            "fcnet_activation": "relu",
            "custom_model" : "fcn",
           "custom_model_config" : {"use_parametric": True,
                                    "env_obs_name": "grid"}
        }
    
    # Configure the environment 

    env_config = {
    "env_name": "rte_case14_realistic",
    "keep_observations": ["rho", "gen_p", "load_p","p_or","p_ex","timestep_overflow",  
                                                                      "maintenance", 
                                                                      "topo_vect"],
    #"keep_actions": ["change_bus", "change_line_status"],
    "keep_actions": ["change_bus"],
    "convert_to_tuple": True, # ignored if act_on_singe or medha_actions
    "act_on_single_substation": True, # ignored if medha = True
    "medha_actions": True,
    "rho_threshold": 0,
    "use_parametric": True 
    }

    #env_glop = grid2op.make(env_config["env_name"],reward_class = L2RPNReward, test=False) 

    # We can now either train directly with RLib trainer or with Ray Tune
    # The latter is preffered for logging and experimentation purposes

    use_tune = False

    if use_tune:
        tune_config = {
        "env": "Grid_Gym",
        "env_config": env_config,  # config to pass to env class,
        "model" : model_config,
        "log_level":"INFO",
        "framework": "torch",
        "lr": tune.grid_search([0.01, 0.001, 0.0001])} # just an example

        analysis = ray.tune.run(
        ppo.PPOTrainer,
        config=tune_config,
        local_dir="/Users/blazejmanczak/Desktop/School/Year 2/Thesis/runPowerNetworks/log_files",
        stop={"training_iteration": 10},
        checkpoint_at_end=True,
        callbacks=[WandbLoggerCallback(
                    project="grid2op",
                    api_key =  WANDB_API_KEY,
                    log_config=True)]
        )
    

    else: # use trainer directly 
    
    # Regular PPO trainer [works]
        print("[INFO]:Using Ray Trainer directly")
        trainer = ppo.PPOTrainer(env=Grid_Gym, config={
        "env_config": env_config,  # config to pass to env class,
        #"env_config": {"env_name":"l2rpn\_case14_sandbox"}, 
        "model" : model_config,
        "log_level":"INFO",
        "framework": "torch",
        "rollout_fragment_length": 16, # 16
            "sgd_minibatch_size": 64, # 64
            "train_batch_size": 512, #2048,
        'num_workers':1,
        "lr" : 1e-3,
        "vf_clip_param": 1000

    })

    # Trying a custom PPO trainer [no effect]

        # print("[INFO]:Using Ray Trainer directly")
        # trainer = CustomPPOTrainer(env=Grid_Gym,
        #     config={
        #             "env_config": env_config,  # config to pass to env class,
        #             #"env_config": {"env_name":"l2rpn\_case14_sandbox"}, 
        #             "model" : model_config,
        #             "log_level":"INFO",
        #             "framework": "torch",
        #             "rollout_fragment_length": 16, # 16
        #                 "sgd_minibatch_size": 64, # 64
        #                 "train_batch_size": 512, #2048,
        #             'num_workers':1,
        #             "lr" : 1e-3,
        #             "vf_clip_param": 1000}
        #         )

        # Trying a custom PPO policy [no effect]
        # print("[INFO]:Using Ray Trainer directly")
        # TrainerWithCustomPolicy = ppo.PPOTrainer.with_updates(
        #                             default_policy = CustomPPOTorchPolicy)
        # trainer = TrainerWithCustomPolicy(env=Grid_Gym,
        #     config={
        #             "env_config": env_config,  # config to pass to env class,
        #             #"env_config": {"env_name":"l2rpn\_case14_sandbox"}, 
        #             "model" : model_config,
        #             "log_level":"INFO",
        #             "framework": "torch",
        #             "rollout_fragment_length": 16, # 16
        #                 "sgd_minibatch_size": 64, # 64
        #                 "train_batch_size": 512, #2048,
        #             'num_workers':1,
        #             "lr" : 1e-3,
        #             "vf_clip_param": 1000}
        #         )

        # and then train it for a given number of iteration
        #trainer.restore("/Users/blazejmanczak/ray_results/PPO_Grid_Gym_2021-11-24_09-43-05pypjh4z5/checkpoint_000091/checkpoint-91")
        for step in range(1000):
            result = trainer.train()
            print(result["episode_len_mean"], flush = True)
            if step % 5 == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)
            print("-"*40, flush = True)
    
    
    
    
    # analysis = ray.tune.run(
    #     ppo.PPOTrainer,
    #     config=tune_config,
    #     local_dir="/Users/blazejmanczak/Desktop/School/Year 2/Thesis/runPowerNetworks/log_files",
    #     stop={"training_iteration": 10},
    #     checkpoint_at_end=True)
         


    #then define a "trainer"
    # trainer = ppo.PPOTrainer(env=Grid_Gym, config={
    #     "env_config": env_config,  # config to pass to env class,
    #     #"env_config": {"env_name":"l2rpn\_case14_sandbox"}, 
    #     "model" : model_config,
    #     "log_level":"INFO",
    #     "framework": "torch",
    #     "rollout_fragment_length": 16,
    #         "sgd_minibatch_size": 64,
    #         "train_batch_size": 2048,

    #     "vf_clip_param": 1000

    # })

    # trainer = ppo.PPOTrainer(env=MyEnv, config={
        # #"env_config": env_config,  # config to pass to env class,
        # "env_config": {"env_name":"rte_case14_realistic"}, 
        # "model" : model_config,
        # "log_level":"INFO",
        # "framework": "torch",
        # "rollout_fragment_length": 16,
        #     "sgd_minibatch_size": 64,
        #     "train_batch_size": 2048,

        # "vf_clip_param": 1000

        # })
    