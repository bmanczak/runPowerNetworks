import logging
import numpy as np
import pandas as pd
import grid2op
import os
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List, Optional, Sequence, Tuple
from collections import defaultdict
from tqdm import tqdm

from grid2op_env.utils import get_sub_id_to_action
from evaluation.utils import get_do_nothing


class AgentInfo:
    """
    Class that stores the agent infromation from run_eval.py
    and performs basic transformation for downstream analysis.
    """
    def __init__(self, actions, chronic_to_num, topo_vects, all_actions = None, do_nothing_actions = None, rewards = None, avg_time_per_step = None):
        self.actions = actions
        self.chronic_to_num = chronic_to_num
        self.topo_vects = topo_vects
        self.all_actions = all_actions
        self.do_nothing_actions = do_nothing_actions
        self.rewards = rewards
        self.avg_time_per_step = avg_time_per_step
    
    @property
    def action_to_sub_id(self):
        """
        Maps the actions to sub
        """
        return get_sub_id_to_action(self.all_actions)
    
    @property
    def mean_chronic_length(self):
        """
        Returns the mean length of the chronic
        """
        if isinstance(list(self.chronic_to_num.values())[0], list):
            return np.mean([steps[-1] for chronic,steps in self.chronic_to_num.items()])
        else: # ensure backward compatibility when the values were ints
            return np.mean(list(self.chronic_to_num.values()))
    @property
    def std_chronic_length(self):
        """
        Returns the mean length of the chronic
        """
        if isinstance(list(self.chronic_to_num.values())[0], list):
            return np.std([steps[-1] for chronic,steps in self.chronic_to_num.items()])
        else: # ensure backward compatibility when the values were ints
            return np.std(list(self.chronic_to_num.values()))

    @property
    def num_imperfect_chronics(self):
        """ Returns the number of imperfect solved chronics."""
        # Get the list with the number of survived steps per chronic
        if isinstance(list(self.chronic_to_num.values())[0], list):
            num_step_chron=  [steps[-1] for chronic,steps in self.chronic_to_num.items()]
        else: # ensure backward compatibility when the values were ints
            num_step_chron = list(self.chronic_to_num.values())

        num_imperfect_chron = len([x for x in num_step_chron if x != 8064])
        
        return num_imperfect_chron


    def quantile_chronic_length(self, quantile = 0.5):
        """
        Returns the given quantile of the chronic length.
        """
        if isinstance(list(self.chronic_to_num.values())[0], list):
            return np.quantile([steps[-1] for chronic,steps in self.chronic_to_num.items()], q = quantile)
        else: # ensure backward compatibility when the values were ints
            return np.quantile(list(self.chronic_to_num.values()), q = quantile)

    @property
    def substation_list(self):
        return list(self.action_to_sub_id.keys())
    

class ActionAnalysis:
    """
    Annalyzes and transforms the data stored in AgentInfo.
    """

    def __init__(self, actions:  Dict[int, List[grid2op.Action.BaseAction]]):
        self.actions = actions

    def order_chronics_per_activity(self):
        """
        Sorts the actions dictionary by the number of actions in the chronic.
        """
        return dict(sorted(self.actions.items(),
                         key=lambda x: len(x[1]), reverse=True))

    def pop_implicit_no_op(self, topo_vects: Dict[int, List[np.array]], chronic_to_num_step =Dict[int, List[int]] )\
         -> Dict[int, List[grid2op.Action.BaseAction]]:
        """
        Removes action that do not change the topology.

        Parameters:
        -----------

        topo_vects: Dict[int, List[np.array]] 
        """
        
        actions_topo_change = defaultdict(list)
        chronic_to_num_topo_change = defaultdict(list)
        for chronic_id, topo_list in topo_vects.items():
            for ix, topo in enumerate(topo_list):
                if ix == 0:
                    actions_topo_change[chronic_id].append(self.actions[chronic_id][0])
                    chronic_to_num_topo_change[chronic_id].append(chronic_to_num_step[chronic_id][0])
                    continue
                #print(ix)
                if not np.array_equal(topo_list[ix-1], topo_list[ix]):
                    # if  self.actions[chronic_id][ix-1].as_dict() == self.actions[chronic_id][ix].as_dict():
                    #     print(f"Chronic {chronic_id} action {ix} is the same as action {ix-1} but the topology is different")
                    try:
                        actions_topo_change[chronic_id].append(self.actions[chronic_id][ix])
                        chronic_to_num_topo_change[chronic_id].append(chronic_to_num_step[chronic_id][ix])
                    except:
                        logging.warning(f"Mismatch in shape of topo_vects and action for chronic {chronic_id}. Skipping action from topo_vect {ix}")
                        
        
        self.actions_topo_change = actions_topo_change
        self.chronic_to_num_topo_change = chronic_to_num_topo_change
 
    @staticmethod
    def action_to_sub_id(actions:  Dict[int, List[grid2op.Action.BaseAction]]) \
            -> Dict[int, List[int]]:
        """
        Transforms the actions dict from: {chronic : list of actions dict}
                                     into {chronic : list of affected sub ids dict}.

        Parameters:
        -----------

        actions: Dict[int, List[grid2op.Action.BaseAction]]
            The action dict.
        """
        grouped = defaultdict(list)
        for k, v in actions.items():
            for act in v:
                if act.as_dict() != {}: # do nothing 
                    modified_subs = act.as_dict()["set_bus_vect"]['modif_subs_id']
                    if len(modified_subs) > 1:
                     logging.warning("More than one substation modified. Choosing the first one.")
                    grouped[k].append(modified_subs[0])
                else:
                    grouped[k].append(-1)
        
        return grouped

    @staticmethod
    def substation_count(grouped: Dict[int, List[int]]) -> Dict[int, int]:
        """
        Given a dictionary with values being a list of affected substation ids
        (output of `action_to_sub_id` method), 
        returns a dictionary with keys being the substation ids
        and values number of actions that changed the substation.

        Parameters:
        -----------
        grouped: Dict[int, List[int]])
            Output of action_to_sub_id method.
        """
        count = defaultdict(int)
        for chronic_sub_list in grouped.values():
            for affected_sub in chronic_sub_list:
                count[affected_sub] += 1
        return count

    @staticmethod
    def count_actions(actions:Dict[int, List[grid2op.Action.BaseAction]]) \
         -> int:
        """
        Counts the number of actions in the actions dict.
        """
        count = 0
        for k, v in actions.items():
            count += len(v)
        return count
    
    @staticmethod
    def get_unique_topos(topo_vects: Dict[int, List[np.array]]) \
         -> int:
        """
        Returns the array of unqiue topologies.
        """
        all_topos = []
        for chronic in topo_vects.keys():
            all_topos.append(topo_vects[chronic])

        return np.unique(np.concatenate(all_topos), axis = 0, return_counts=True)

    @property
    def chronic_count(self):
        return len(self.actions)


def get_analysis_objects(eval_path: str, wandb_run_id:str = None):
        """ 
        Given a path to the eval results, returns the action analysis and agent info objects.

        Parameters:
        -----------

        eval_path: str
            Path to the eval results obtained by run_eval.py.
        wandb_run_id: Optional[str]
            If not None, will use this to fetch the env_config from wandb.
            Otherwise an automatic parsing will be performed.
        
        """
        if len(os.listdir(eval_path)) == 3: # backward compatibility
                running_deprecated_eval = True
                action_path, chronic_to_num_path, topo_vects_path  = sorted(os.listdir(eval_path), key = lambda x: x[0])

                actions, chronic_to_num, topo_vects = np.load(os.path.join(eval_path, action_path), allow_pickle=True).item(), \
                                                np.load(os.path.join(eval_path, chronic_to_num_path), allow_pickle=True).item(), \
                                                np.load(os.path.join(eval_path, topo_vects_path), allow_pickle=True).item()
        else:
                running_deprecated_eval = False
                actions, topo_vects, chronic_to_num, rewards, avg_time_per_step  = np.load(os.path.join(eval_path, "actions.npy"), allow_pickle=True).item(), \
                                                np.load(os.path.join(eval_path, "topo_vects.npy"), allow_pickle=True).item(), \
                                                np.load(os.path.join(eval_path, "chronic_to_num_steps.npy"), allow_pickle=True).item(), \
                                                np.load(os.path.join(eval_path, "rewards.npy"), allow_pickle=True).item(), \
                                                np.load(os.path.join(eval_path, "avg_time_per_step.npy"), allow_pickle=True).item()

        
        if wandb_run_id is None:
                wandb_run_id = "_".join(eval_path.split("_num_workers")[0].split("_")[-3:-1])#"_".join(EVAL_RESULTS_PATH.split("Grid_Gym_")[1].split("_")[0:2])

        do_nothing_actions, all_actions = get_do_nothing(wandb_run_id = wandb_run_id)
        
        agent_info = AgentInfo(actions, chronic_to_num, topo_vects, all_actions, do_nothing_actions, rewards, avg_time_per_step)
        
        logging.info(f"Mean chronic length {agent_info.mean_chronic_length}")
        logging.info(f"0.2 quantile chronic length {agent_info.quantile_chronic_length(0.25)}")
        action_analysis = ActionAnalysis(actions)

        if running_deprecated_eval:
                return agent_info, action_analysis, topo_vects, None, None
        else: 
                return agent_info, action_analysis, topo_vects, rewards, avg_time_per_step

def get_evaluation_data(eval_path: str, wandb_run_id:Optional[str] = None):
        """
        Given the path to the eval results, returns processed objects needed to plot the results.
        Works on the output of get_analysis_objects method.

        Parameters:
        -----------

        eval_path: str
            Path to the eval results obtained by run_eval.py.
        wandb_run_id: Optional[str]
            If not None, will use this to fetch the env_config from wandb.
        """

        agent_info, action_analysis, topo_vects, rewards, avg_time_per_step = get_analysis_objects(eval_path, wandb_run_id)
        action_analysis.pop_implicit_no_op(topo_vects, agent_info.chronic_to_num)

        grouped = action_analysis.action_to_sub_id(action_analysis.actions) # needed for explicit do nothing
        grouped_topo_change = action_analysis.action_to_sub_id(action_analysis.actions_topo_change)

        sub_num_actions = action_analysis.substation_count(grouped_topo_change)
        sub_num_actions_all = action_analysis.substation_count(grouped)
        
        # Replace "Implicit No Op" with key "10000" for easier sorting by keys later
        sub_num_actions["10000"] = abs(sub_num_actions_all["0"] - \
                (action_analysis.count_actions(action_analysis.actions) -  action_analysis.count_actions(action_analysis.actions_topo_change)))

        # Add the substations without any actions
        for sub in agent_info.substation_list:
                if str(sub) not in sub_num_actions:
                        sub_num_actions[str(sub)] = 0
        # Replace "Explicit No Op" with key "20000" for easier sorting by keys later
        sub_num_actions["20000"] = sub_num_actions_all["0"]
        sub_num_actions.pop("0", None)

        sub_num_actions = dict(sorted(sub_num_actions.items(), key = lambda x: float(x[0])  ))

        # Bring back the proper key names
        sub_num_actions["Implict No op"] = sub_num_actions["10000"] 
        sub_num_actions["Explicit No op"] = sub_num_actions["20000"]  

        sub_num_actions.pop("10000", None)
        sub_num_actions.pop("20000", None)

        num_actions = sum(sub_num_actions.values())
        normalized_sub_num_actions = {key: value/num_actions for key, value in sub_num_actions.items()}

        leave_keys = ['1', '2', '3', '4', '5', '8', '12', 'Implict No op', 'Explicit No op']
        for key, val in sub_num_actions.items():
                if key not in leave_keys:
                        normalized_sub_num_actions.pop(key, None)

        logging.info(f"Normalized substation actions {normalized_sub_num_actions}")
        return normalized_sub_num_actions, num_actions, action_analysis.actions_topo_change, agent_info, action_analysis

def process_eval_data_multiple_agents(eval_paths: Dict[str, Tuple[str, str]],
    data_per_algorithm: Optional[Dict] = None) \
    -> Dict :
    """
    Given a dictionary {algo_name : (eval_path, wandb_run_id)} returns a nested dictionary  
    with summary statistics for each agent.

    Parameters:
    -----------

    eval_paths: Dict[str, Tuple(str, str)]
        Dictionary of structure {algo_name : (eval_path, wandb_run_id)} 
    data_per_algorithm: Optional[Dict]
        If not None, will not process algorithms with algo_name in this dictionary.
    """
    data_per_algorithm = {} if data_per_algorithm is None else data_per_algorithm

    for name, path in eval_paths.items():
        if name not in data_per_algorithm:
            if isinstance(path, tuple):
                path, wandb_run_id = path
            else: 
                wandb_run_id = None

            logging.info(f"Processing name {name} ")

            normalized_sub_num_actions, num_actions, actions_topo_change,  agent_info, action_analysis = get_evaluation_data(path,wandb_run_id )
           
            data_per_algorithm[name] = {} 
            data_per_algorithm[name]["action_distr"] = normalized_sub_num_actions

            data_per_algorithm[name]["num_actions"] = num_actions
            data_per_algorithm[name]["actions_topo_change"] = actions_topo_change

            data_per_algorithm[name]["agent_info"] = agent_info
            data_per_algorithm[name]["action_analysis"] = action_analysis
        else:
            continue
    
    return data_per_algorithm


def compile_table_df(data_per_algorithm:Dict) -> pd.DataFrame:
    """
    Given the output of process_eval_data_multiple_agents constructs
    a data frame that summarizes the results.
    """

    df_dict = {}

    for name, value in data_per_algorithm.items():
        df_dict[name] = {}

        for key, val in value.items():
            # Skip the keys with intricate objects
            if key in ["actions_topo_change", "agent_info", "action_analysis"]:
                continue
            # Process the 1-D nested dictionaries
            if type(val) == dict: 
                for sub_num, prop_actions in val.items():
                    df_dict[name][sub_num] = prop_actions
            else:
                df_dict[name][key] = val

        df_dict[name]["num_topo_actions"] = df_dict[name]["num_actions"] * (1 - df_dict[name]["Implict No op"] - df_dict[name]["Explicit No op"] )  # actions that change the topology
        
        # Ensure backward compatibility
        if isinstance(data_per_algorithm[name]["agent_info"].mean_chronic_length, list):
            df_dict[name]["mean_chronic_length"] = data_per_algorithm[name]["agent_info"].mean_chronic_length[0]

        # Add metrics that do not require extrac compute
        df_dict[name]["mean_chronic_length"] = data_per_algorithm[name]["agent_info"].mean_chronic_length
        df_dict[name]["std_chronic_length"] = data_per_algorithm[name]["agent_info"].std_chronic_length
        df_dict[name]["mean_normalized_reward"] = (np.mean(list(data_per_algorithm[name]["agent_info"].rewards.values())) / df_dict[name]["mean_chronic_length"] ) * 8064
        
        df_dict[name]["0.05th_quantile "] = data_per_algorithm[name]["agent_info"].quantile_chronic_length(0.05)
        df_dict[name]["median_chronic_length"] = data_per_algorithm[name]["agent_info"].quantile_chronic_length(0.5)
        df_dict[name]["num_imperfect_chron"] = data_per_algorithm[name]["agent_info"].num_imperfect_chronics

    # Dict to DataFrame
    df = pd.DataFrame.from_dict(df_dict, orient='index')#.reset_index()
    df["No Op"] = df["Explicit No op"] + df["Implict No op"]

    for name, value in data_per_algorithm.items():
        df_dict[name] = {}
        # Number of unique actions
        all_topos = np.concatenate(list(data_per_algorithm[name]["agent_info"].topo_vects.values()))
        unique_topos, topo_count  = np.unique(all_topos, axis=0, return_counts=True )
        everything_disconnected_mask = (unique_topos>=1).any(axis=1) # filter everything disconneced case
        appear_more_than_once_mask = topo_count > 1 # filter the just before collapse topologies
        # Filter out the disconnected lines 
        df.loc[name, "num_unique_topos"] = unique_topos[everything_disconnected_mask & appear_more_than_once_mask].shape[0]
    
        # Topological depth
        mean_topo_depth = defaultdict(list)
        topo_vects = data_per_algorithm[name]["agent_info"].topo_vects
        for _, topo_vects_chronic in topo_vects.items():
            # Do not take disconnected lines into the depth calculation
            diff_from_default = np.array(topo_vects_chronic) - 1
            diff_from_default_ignore_disconnected = np.where(diff_from_default > 0, diff_from_default, 0)       
            mean_topo_depth[name].append(np.mean
                                        (np.sum(
                                                diff_from_default_ignore_disconnected, axis = 1)
                                                )
                )
        df.loc[name, "mean_topo_depth"] = np.mean(mean_topo_depth[name])
        df.loc[name, "std_topo_depth"] = np.std(mean_topo_depth[name])

        # Action sequences 

        out = get_action_sequences(name, data_per_algorithm)
        df.loc[name, "mean_repeat_seq"] = np.mean(list(out.values()) )
        df.loc[name, "std_repeat_seq"] = np.std(list(out.values()))
        df.loc[name, "mean_seq_len"] = np.mean([len(key) for key, val in out.items() if val > 1])
        df.loc[name, "std_seq_len"] = np.std([len(key) for key, val in out.items() if val > 1])
        df.loc[name, "num_unqiue_seq"] = len(out)

    return df



def get_action_sequences(algo_type : str, data_per_algorithm: Dict) \
    -> Dict[Tuple[List], int]:
    """
   Given the data_per_algorithm, returns a dicttionary with 
   {(action_sequence) : num_occurences}
   
   Parameters:
    -----------
    algo_type: str
        Name of the algorithm
    data_per_algorithm: Dict
        Output of process_eval_data_multiple_agents function.
   """

    def process_single_chronic(chronic_to_num_arr):
        """
        Returns a nested list of indicies of the actions that are part of the same sequence.
        
        """
        # Calcualte the difference in topologies between steps
        time_step_diff = np.diff(chronic_to_num_arr) == 1
        # Bring shape back the orignal one
        concat_val = True if time_step_diff[0] else False
        time_step_diff_match_dim = np.concatenate(([concat_val], time_step_diff))
        
        # Seperate different sequences
        sequences = []
        temp_seq = []
        # new_seq = False
        for i, boolean in enumerate(time_step_diff_match_dim):
            if boolean:
                if len(temp_seq) == 0 and i > 0: # add the action that starts the sequence
                    temp_seq.append(chronic_to_num_arr[i-1])
                temp_seq.append(chronic_to_num_arr[i])

            else: # if not boolean:
                # new_seq = True
                if len(temp_seq) > 0:
                    sequences.append(temp_seq)
                    temp_seq = []
        
        return sequences

    sequence_dict = defaultdict(int) # key: a tuple of topological vectors (tuples), values: count
    count_chrons = 0
    count_sequences = 0

    for chronic_id in tqdm(data_per_algorithm[algo_type]["agent_info"].chronic_to_num.keys()):
    # for chronic_id in tqdm([2]):
        chronic_to_num_arr = [elem -1 for elem in data_per_algorithm[algo_type]["action_analysis"].chronic_to_num_topo_change[chronic_id]]  #.pop("Greedy") #np.array([1,2,4,5, 7,8,9, 13, 20,21, 1998, 2002])
        if len(chronic_to_num_arr) < 2:
            continue
        count_chrons += 1
        sequences = process_single_chronic(chronic_to_num_arr)
        count_sequences += len(sequences)
        temp_seq = []

        chron_to_topo_dict = dict(zip(
            data_per_algorithm[algo_type]["agent_info"].chronic_to_num[chronic_id][:-1], data_per_algorithm[algo_type]["agent_info"].topo_vects[chronic_id]))
        # Match the keys of chron_to_topo_dict and chronic_to_num_arr by shifting the keys by one to the left
        chron_to_topo_dict = {key-1: chron_to_topo_dict[key] for key in chron_to_topo_dict.keys()}
         
        for seq in sequences:
            temp_seq = []
            for timestep in seq:
        
                temp_seq.append(tuple(chron_to_topo_dict[timestep]))
                if len(temp_seq) > 1:
                    assert temp_seq[-1] != temp_seq[-2]
            assert len(temp_seq) == len(seq)

            if tuple(temp_seq) in sequence_dict:
                sequence_dict[tuple(temp_seq)] += 1
            else:
                sequence_dict[tuple(temp_seq)] = 1
        
    assert count_sequences == sum(sequence_dict.values()), "Count of sequences does not match the count in the dictionary"
    sequence_dict = dict(sorted(sequence_dict.items(),
                         key=lambda x: (x[1],len(x[0])), reverse=True))
    return sequence_dict