# gym specific, we simply do a copy paste of what we did in the previous cells, wrapping it in the
# MyEnv class, and train a Proximal Policy Optimisation based agent
import os
import ray
import logging
import wandb


from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.agents import ppo  # import the type of agents
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

from ray import tune
from ray.tune.registry import register_env
from ray.tune.integration.wandb import WandbLoggerCallback

from dotenv import load_dotenv # security keys

from models.mlp import SimpleMlp
from grid2op_env.grid_to_gym import Grid_Gym


load_dotenv()
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

# from custom_trainer import CustomPPOTrainer
# from custom_policy import CustomPPOTorchPolicy


logging.basicConfig(format='[INFO]: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)



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

    # We can now either train directly with RLib trainer or with Ray Tune
    # The latter is preffered for logging and experimentation purposes

    use_tune = False

    if use_tune:
        wandb.init(project="grid2op_rlib", entity="bmanczak")
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
    