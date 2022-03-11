import logging
import numpy as np
import pandas as pd
import grid2op
import os
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List, Optional, Sequence
from collections import defaultdict

from grid2op_env.utils import get_sub_id_to_action
from evaluation.utils import get_do_nothing


class AgentInfo:
    """
    Parses and holds the evaluation information about the agent.
    """
    def __init__(self, actions , chronic_to_num, topo_vects,
                 all_actions = None, do_nothing_actions = None):
        """
        Initialized with the output of run_eval.py
        """
        self.actions = actions
        self.chronic_to_num = chronic_to_num
        self.topo_vects = topo_vects
        self.all_actions = all_actions
        self.do_nothing_actions = do_nothing_actions
    
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
    def substation_list(self):
        return list(self.action_to_sub_id.keys())
    

class ActionAnalysis:

    def __init__(self, actions:  Dict[int, List[grid2op.Action.BaseAction]]):
        self.actions = actions

    def order_chronics_per_activity(self):
        return dict(sorted(self.actions.items(),
                         key=lambda x: len(x[1]), reverse=True))

    def pop_implicit_no_op(self, topo_vects: Dict[int, List[np.array]] )\
         -> Dict[int, List[grid2op.Action.BaseAction]]:
        """
        Removes action that do not change the topology.

        Parameters:
        -----------

        topo_vects: Dict[int, List[np.array]] 
        """
        
        actions_topo_change = defaultdict(list)
        for chronic_id, topo_list in topo_vects.items():
            for ix, topo in enumerate(topo_list):
                if ix == 0:
                    actions_topo_change[chronic_id].append(self.actions[chronic_id][0])
                    continue
                #print(ix)
                if not np.array_equal(topo_list[ix-1], topo_list[ix]):
                    # if  self.actions[chronic_id][ix-1].as_dict() == self.actions[chronic_id][ix].as_dict():
                    #     print(f"Chronic {chronic_id} action {ix} is the same as action {ix-1} but the topology is different")
                    try:
                        actions_topo_change[chronic_id].append(self.actions[chronic_id][ix])
                    except:
                        print(f"Mismatch in shape of topo_vects and action for chronic {chronic_id}. Skipping action from topo_vect {ix}")
                        
        
        self.actions_topo_change = actions_topo_change
 
    @staticmethod
    def action_to_sub_id(actions:  Dict[int, List[grid2op.Action.BaseAction]]) \
            -> Dict[int, List[int]]:
        """
        Transforms the action in the chronic : list of actions dict
        into chronic : list of affected sub ids dict.

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
        count = 0
        for k, v in actions.items():
            count += len(v)
        return count
    
    @staticmethod
    def get_unique_topos(topo_vects: Dict[int, List[np.array]]) \
         -> int:
        all_topos = []
        for chronic in topo_vects.keys():
            all_topos.append(topo_vects[chronic])

        return np.unique(np.concatenate(all_topos), axis = 0, return_counts=True)

    @property
    def chronic_count(self):
        return len(self.actions)


def get_analysis_objects(eval_path: str, wandb_run_id:Optional[str] = None):

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
        action_path, chronic_to_num_path, topo_vects_path  = sorted(os.listdir(eval_path), key = lambda x: x[0])

        actions, chronic_to_num, topo_vects = np.load(os.path.join(eval_path, action_path), allow_pickle=True).item(), \
                                        np.load(os.path.join(eval_path, chronic_to_num_path), allow_pickle=True).item(), \
                                        np.load(os.path.join(eval_path, topo_vects_path), allow_pickle=True).item()
        
        if wandb_run_id is None:
                wandb_run_id = "_".join(eval_path.split("_num_workers")[0].split("_")[-3:-1])

        do_nothing_actions, all_actions = get_do_nothing(wandb_run_id = wandb_run_id)
        
        agent_info = AgentInfo(actions, chronic_to_num, topo_vects, all_actions, do_nothing_actions)
        
        print(f"Mean chronic length {agent_info.mean_chronic_length}")
        action_analysis = ActionAnalysis(actions)

        return agent_info, action_analysis, topo_vects
    
def get_evaluation_data(eval_path: str, wandb_run_id:str = None):
        
        """
        Processes
        """
        agent_info, action_analysis, topo_vects = get_analysis_objects(eval_path, wandb_run_id)
        action_analysis.pop_implicit_no_op(topo_vects) # instantiates action_analysis.actions_topo_change

        # Instead of mapping of chronic to action objects, get map of chronics to affected substations
        grouped = action_analysis.action_to_sub_id(action_analysis.actions) # needed for explicit do nothing
        grouped_topo_change = action_analysis.action_to_sub_id(action_analysis.actions_topo_change)

        # Get the dictionary with the number of actions that changed each substation
        sub_num_actions = action_analysis.substation_count(grouped_topo_change)
        sub_num_actions_all = action_analysis.substation_count(grouped)

        # Use 10000 as the implicit no op action (so it's possible to sort later)
        sub_num_actions["10000"] = abs(sub_num_actions_all["0"] - \
                (action_analysis.count_actions(action_analysis.actions) -  action_analysis.count_actions(action_analysis.actions_topo_change)))

        # Add the substations without any actions
        for sub in agent_info.substation_list:
                if str(sub) not in sub_num_actions:
                        sub_num_actions[str(sub)] = 0

        # Use 20000 as the explicit no op action (so it's possible to sort later)
        sub_num_actions["20000"] = sub_num_actions_all["0"]
        sub_num_actions.pop("0", None) # Delete the implicit no op action

        sub_num_actions = dict(sorted(sub_num_actions.items(), key = lambda x: float(x[0])  ))

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

        print(f"Normalized substation actions {normalized_sub_num_actions}")
        return normalized_sub_num_actions, num_actions, action_analysis.actions_topo_change, agent_info, action_analysis
