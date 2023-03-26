# Hierarchical Reinforcement Learning for Power Network Topology Control

This repository contains the code for experiments conducted for my master thesis realized at Amsterdam Machine Learning Lab and TenneT.

## Paper

Please see the paper **Hierarchical Reinforcement Learning for Power Network Topology Control** (link coming soon) for more details. If this code is useful please cite our paper:

*BibTeX coming soon*

### Dependencies 

The `env.yml` file can be used to install the dependencies. 

### Data setup

We run our experiments with the `rte_case14_realistic` modeled in [Grid2Op](https://grid2op.readthedocs.io/en/latest/quickstart.html) package. The 1000 chronics (=episodes) are divided into a train, validation, and test set. The chronic numbers that we use for each split are saved in NumPy arrays in the `grid2op_env/train_val_test_split` folder.

The following has to be run only once to download and set up the environment:

```
import grid2op

env_name = "rte_case14_realistic"
env = grid2op.make(env_name)

val_chron, test_chron = np.load("grid2op_env/train_val_test_split/val_chronics.npy"), \
 np.load("/grid2op_env/train_val_test_split/test_chronics.npy")

nm_env_train, m_env_val, nm_env_test = env.train_val_split(test_scen_id=test_chron, # last 10 in test set
 add_for_test="test",
 val_scen_id=val_chron, # last 20 to last 10 in val test
 )

env_train = grid2op.make(env_name+"_train")
env_val = grid2op.make(env_name+"_val")
env_test = grid2op.make(env_name+"_test")

```
### Agent training

Note that wandb is used for monitoring the progress of the experiment.
If you wish to use wandb make sure to specify the `WANDB_API_KEY` in the `.env` file. Alternatively, comment out `WandbLoggerCallback` in the `train.py` file.

#### Setup

We train and benchmark the models in an environment with and without outages. The environment setting is controlled by the boolean `--with_opponennt` keyword argument in the `train.py` script.

By default, the 5 best checkpoints in terms of mean episode reward will be saved in the `log_files` directory.

#### Native and hybrid agents

These agents support training with *PPO* and *SAC* algorithms. 
To train these agents, go to the `main` branch and run the `train.py` file with desired keyword arguments. The choice of hyperparameters in a `.yaml` file. Specifications used in the paper are found in the `experiments` folder under the corresponding algorithm name.

For instance, to train a hybrid PPO agent in the setting with outages for 1000 iterations and over 10 different seeds run:

``` 
python train.py --algorithm ppo \
 --algorithm_config_path experiments/ppo/ppo_run_threshold.yaml \
 --with_opponent True \
 --num_iters 1000 \
 --num_samples 10 \
 ```

See the `argparse` help for more details on keyword arguments.

#### Hierarchical agent


To train a fully hierarchical agent go to the `hierarchical_approach` branch and run the `train_hierarchical.py` file with desired keyword arguments. Similar to the native and hybrid agents, the choice of hyperparameters in a `.yaml` file.

To train the hierarchical agent in the setting with outages for 1000 iterations and over 10 different seeds run:

```
python train_hierarchical.py --algorithm ppo \
 --algorithm_config_path experiments/hierarchical/full_mlp_share_critic.yaml \
 --use_tune True \
 --num_iters 1000 \
 --num_samples 16 \
 --checkpoint_freq 10 \
 --with_opponent True 
```


### Evaluation

To run the trained agent on the set of test chronics run:

```
python evaluation/run_eval.py --agent_type X \
 --checkpoint_path Y \
 --checkpoint_num Z \
 --use_split test \
```
If the agent being evaluated is a fully hierarchical (i.e. non-hybrid) add keyword argument `--hierarchical True`.
Except for printing the mean episode length, this script involves data collection that is needed for further analysis. The data is saved in a folder `evaluation/eval_results` and can be used for further analysis.

The functionality for further analysis is implemented in the `evaluation/results_analysis.py` file. Given the path to evaluation results it is easy to obtain a table with the statistics:

```
from evaluation.result_analysis import process_eval_data_multiple_agents, \
 get_analysis_objects, \
 compile_table_df

EVAL_PATHS = {"Agent Type 1": (path_to_eval_results, "wandb_num"),
 "Agent Type 2": (path_to_eval_results, "wandb_num"), ...}

data_per_algorithm = process_eval_data_multiple_agents(EVAL_PATHS)

# Compile the data frame from which we will later plot the results
df = compile_table_df(data_per_algorithm)
```
### Repository overview 

`evaluation` contains the code for benchmarking trained agents

`experiments` contains the specification of the model hyperparameters and custom callbacks 

`grid2op_env` contains the environment wrappers, train/test/val split, and data used to scale the observations

`models` contains the code for the torch models used in the experiments

`notebooks` contains miscellaneous notebooks used in the course of development and evaluation. Notably `sub_node_model.ipnyb` contains an alpha version of a Graph Neural Network (GNN) based policy.
 
