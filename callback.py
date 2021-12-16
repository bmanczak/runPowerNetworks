import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import io
import logging
import torch
import time
import sys

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.logger import TBXLogger
from ray.util.debug import log_once
from ray.tune.result import (TRAINING_ITERATION, TIME_TOTAL_S, TIMESTEPS_TOTAL)
from ray.util.ml_utils.dict import flatten_dict

from tensorboardX import SummaryWriter
from typing import TYPE_CHECKING, Dict, List, Optional, TextIO, Type

if TYPE_CHECKING:
    from ray.tune.trial import Trial  # noqa: F401

logger = logging.getLogger(__name__)
VALID_SUMMARY_TYPES = [int, float, np.float32, np.float64, np.int32, np.int64] # from rllib

# Plotting settings
matplotlib.rcParams["figure.dpi"] = 200
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
plt.figure(figsize=(14,10), tight_layout=True)


class LogDistributionsCallback(DefaultCallbacks):
    """
    Custom callback to log extra information about
    the distribution of observations and actions.

    Data saved here is later accessed by Custom Tensorboar logger.

    To use it in the experiment, add the following to the tune config:
    "callbacks": LogDistributions, ...

    Assumes that the raw observation has the "grid" attribute that
    stores all the observations generated by the grid2op environment.

    """

   
    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                        result: dict, **kwargs) -> None:
        
        """
        Log the action distribution and extra information about the observation.
        Note that everything in result[something] is logged by Ray.
        """

        if torch.is_tensor(train_batch["obs"]):
            train_batch_obs = train_batch["obs"].numpy()
            train_batch_new_obs = train_batch["new_obs"].numpy()
            train_batch_actions = train_batch["actions"].numpy()
        else:
            train_batch_obs = train_batch["obs"]
            train_batch_new_obs = train_batch["new_obs"]
            train_batch_actions = train_batch["actions"]

      
        changed_topo_vec = np.any(train_batch_new_obs[:, -56:]!=train_batch_obs[:, -56:], axis = -1)
        num_non_zero_actions = np.sum(changed_topo_vec)
        
        # print("Chanhed topo vec shape", changed_topo_vec.shape)
        # print("new obs shape", train_batch["new_obs"].shape)
        # print("changed_topo_vec", torch.sum(changed_topo_vec))
        # if torch.is_tensor(train_batch["actions"]): # for ppo it's numpy, for sac it's tensor
        #     train_batch["actions"] = train_batch["actions"].numpy()
        #num_non_zero_actions = np.sum(train_batch["actions"] != 0)

        # Log the proportion of actions that do not change the topology
        result["prop_topo_action_change"] = num_non_zero_actions/train_batch["actions"].shape[0]
        result["prop_explicit_do_nothing"] = np.sum(train_batch_actions == 0)/num_non_zero_actions
        # Count all of the actions and save them to action
        unique, counts = np.unique(train_batch_actions[changed_topo_vec], return_counts=True)
        action_distr_dic = {}
        for action in range(policy.action_space.n): # initalize the dictionary with all actions
            action_distr_dic[str(action)] = 0
        for action, count in zip(unique, counts):
            action_distr_dic[str(action)] = count/num_non_zero_actions # action distr in percentage
        
        del action_distr_dic[str(0)] # remove the do-nothing action from the action distr
        result["action_distr"] = action_distr_dic
        result["num_non_zero_actions_tried"] = sum([1 for val in action_distr_dic.values() if val > 0])
        #print("time it took to count actions", time.time() - start_time)
class CustomTBXLogger(TBXLogger):
    """
    Custom TBX logger that logs the action distribution and extra information about the actions
    taken by the agent.

    This logger relies on the information logged by the LogDistributionsCallback.
    To use it, add an argument to tune.run() with the following: loggers = [CustomTBX]
    """

    def on_result(self, result: Dict):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        tmp = result.copy()
        for k in [
                "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
        ]:
            if k in tmp:
                del tmp[k]  # not useful to log these
        
        # print("The size of the dictionary is {} bytes".format(sys.getsizeof(result)))
        flat_result = flatten_dict(tmp, delimiter="/")

        # Addition start   
        #start_time = time.time()
        action_distr_dic = result["info"]["learner"]["default_policy"]["custom_metrics"]["action_distr"]
        bar_arr = plot_to_array(bar_graph_from_dict(action_distr_dic))
        self._custom_file_writer = SummaryWriter(self.logdir, flush_secs=30) # this results in many tf.event files
        self._custom_file_writer.add_image("Action_distribution",bar_arr, step, dataformats = "HWC")
        self._custom_file_writer.close()
        #print("Time taken to plot action distribution:", time.time() - start_time)
        # Additon end
        
        path = ["ray", "tune"]
        valid_result = {}
        #start_time = time.time()
        for attr, value in flat_result.items():
            full_attr = "/".join(path + [attr])
            if (isinstance(value, tuple(VALID_SUMMARY_TYPES))
                    and not np.isnan(value)):
                valid_result[full_attr] = value
                self._file_writer.add_scalar(
                    full_attr, value, global_step=step)
            elif ((isinstance(value, list) and len(value) > 0)
                  or (isinstance(value, np.ndarray) and value.size > 0)):
                valid_result[full_attr] = value

                # Must be video
                if isinstance(value, np.ndarray) and value.ndim == 5:
                    self._file_writer.add_video(
                        full_attr, value, global_step=step, fps=20)
                    continue

                try:
                    self._file_writer.add_histogram(
                        full_attr, value, global_step=step)
                # In case TensorboardX still doesn't think it's a valid value
                # (e.g. `[[]]`), warn and move on.
                except (ValueError, TypeError):
                    if log_once("invalid_tbx_value"):
                        logger.warning(
                            "You are trying to log an invalid value ({}={}) "
                            "via {}!".format(full_attr, value,
                                             type(self).__name__))
        #print("Time taken to log:", time.time() - start_time)
        self.last_result = valid_result
        self._file_writer.flush()

# Utility plotting function
def bar_graph_from_dict(dic):
    """
    Given a dictionary, return matplotlib bar graph figure.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(*zip(*dic.items()))
    ax.set_xticklabels(dic.keys(), rotation=90, fontsize = 4)
    ax.set_xlabel('Action')
    ax.set_ylabel('Proportion of all non-zero actions')
    fig.canvas.draw()

    return fig

def plot_to_array(fig):
    """
    Transfrom the matplotlib figure to a numpy array.
    """
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im[:, :, :3] # skip the a in rgba
    return im