# gym specific, we simply do a copy paste of what we did in the previous cells, wrapping it in the
# MyEnv class, and train a Proximal Policy Optimisation based agent
from typing import OrderedDict
import gym
import ray
import gym
import numpy as np
import grid2op 

import torch 
import torch.nn as nn
import logging
import wandb

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

# nice article about Ray https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5

#wandb.init(project="grid2op_rlib", entity="bmanczak")



class Grid_Gym(gym.Env):
    def __init__(self, env_config):
        
        self.env_gym = create_gym_env(env_name = env_config["env_name"],
                                        keep_obseravations= env_config["keep_observations"],
                                        keep_actions= env_config["keep_actions"],
                                        convert_to_tuple=env_config["convert_to_tuple"])
        
        # First serialise as dict, then convert to Dict gym space
        # for the sake of compatibility with Ray Tune
        d = {k: v for k, v in self.env_gym.observation_space.items()}
        self.observation_space = gym.spaces.Dict(d)

        a = {k: v for k, v in self.env_gym.action_space.items()}
        self.action_space = gym.spaces.Dict(a)

    def reset(self):
        obs = self.env_gym.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env_gym.step(action)
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

        activation = model_config.get("fcnet_activation")
        no_final_linear = model_config.get("no_final_linear") # if True, last layer will have out_size=num_outputs
        self.vf_share_layers = model_config.get("vf_share_layers")
      

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers += [nn.Linear(prev_layer_size, size), nn.ReLU(inplace=True)]
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers += [nn.Linear(prev_layer_size, num_outputs), nn.ReLU(inplace=True)]
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers += [nn.Linear(prev_layer_size, hiddens[-1]), nn.ReLU(inplace=True)]
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = nn.Linear(prev_layer_size, num_outputs)
            else:
                self.num_outputs = (
                    [int(np.product(obs_space.shape))] + hiddens[-1:])[-1]


        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
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
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else \
            self._features
       
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
            "custom_model_config" : {}
        }
    
    # Configure the environment 

    env_config = {
    "env_name": "rte_case14_realistic",
    "keep_observations": ["rho", "gen_p", "load_p","p_or","p_ex","timestep_overflow",  
                                                                      "maintenance", 
                                                                      "topo_vect"],
    "keep_actions": ["change_bus", "change_line_status"],
    "convert_to_tuple": True
    }

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
        checkpoint_at_end=True)
    

    else: # use trainer directly 
    

        trainer = ppo.PPOTrainer(env=Grid_Gym, config={
        "env_config": env_config,  # config to pass to env class,
        #"env_config": {"env_name":"l2rpn\_case14_sandbox"}, 
        "model" : model_config,
        "log_level":"INFO",
        "framework": "torch",
        "rollout_fragment_length": 16,
            "sgd_minibatch_size": 64,
            "train_batch_size": 2048,

        "vf_clip_param": 1000

    })

        # and then train it for a given number of iteration
        for step in range(100):
            result = trainer.train()
            print(result["episode_len_mean"], flush = True)
            if step % 5 == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)
            print("-"*40, flush = True)
    
    #env_glop = grid2op.make(env_config["env_name"],reward_class = L2RPNReward, test=False) 
    
    
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
    