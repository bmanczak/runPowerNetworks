import grid2op
import numpy as np

from grid2op.Agent.greedyAgent import GreedyAgent
from grid2op.dtypes import dt_float
from typing import Sequence, List, Dict

class ClassicGreedyAgent(GreedyAgent):
    """
    Greedy agent that will iterate through all the actions
    and choose the one that brings down the max line capacity.
    """

    def __init__(self, action_space: Sequence[grid2op.Action.TopologyAction], action_list: List[grid2op.Action.BaseAction],
                do_nothing_idx: List[int]):
        """
        Parameters:
        ----------
        
            action_space: Sequence[grid2op.Action.TopologyAction]
                 The action space. Needed for super init.
            action_list: action_list: List[grid2op.Action.BaseAction]
                List of actions to simulate by the agent.
            do_nothing_idx: List[int]
                List of indices of the action that do nothing.
             
        """
        GreedyAgent.__init__(self, action_space)
        self.tested_action = action_list
        self.do_nothing_idx = do_nothing_idx[0]
    
    def _get_tested_action(self):
        return self.tested_action 
    
    def act(self, observation: grid2op.Observation.CompleteObservation) \
            -> grid2op.Action.BaseAction:
        """
        Simulates the actions from the list given upon init. and chooses the one
        that brings down the max line capacity.

        Parameters:
        ----------
        observation: : grid2op.Observation.CompleteObservation
            The current observation of the :class:`grid2op.Environment.Environment`

        Returns
        -------
        res: grid2op.Action.Action
            The action that brings the rho the most.
        """
    
        self.resulting_max_rho = np.full(shape=len(self.tested_action), fill_value=np.NaN, dtype=dt_float)

        simul_obs, _, simul_has_error, simul_info = observation.simulate(self.tested_action[self.do_nothing_idx])
        do_nothing_rho = simul_obs.rho.max() if not simul_has_error else float("inf")
        
        for i, action in enumerate(self.tested_action):
            simul_obs, _, simul_has_error, simul_info = observation.simulate(action)

            self.resulting_max_rho[i] = simul_obs.rho.max() if not simul_has_error else float("inf")

        min_rho_idx = int(np.argmin(self.resulting_max_rho))
        best_action = self.tested_action[min_rho_idx]

        if self.resulting_max_rho[min_rho_idx] < do_nothing_rho:
            return best_action
        else:
            return self.tested_action[self.do_nothing_idx]


class RoutingTopologyGreedy(GreedyAgent):
    """
    Greedy agent that considers all the actions for a given
    substation and chooses the one that will minimize the max
    line capacity.

    It will choose among:
      - doing nothing
      - changing the topology of one substation.

    """
    def __init__(self, action_space: Sequence[grid2op.Action.TopologyAction] ,
                 sub_id_to_action_dict: Dict[int, List[grid2op.Action.BaseAction]]) :
        """
        Parameters:
        ----------
        
            action_space: Sequence[grid2op.Action.TopologyAction]
                 The avaliable actions.
            sub_id_to_action_dict: dict
                Mapping between the substation id and
            num_actions ([type], optional): [description]. Defaults to None.
        """
        GreedyAgent.__init__(self, action_space)
        self.tested_action = None
        
        self.sub_id_to_action = sub_id_to_action_dict

    
    def get_num_to_sub(self):
        """
        In case we have less actions than substations we must map the action
        integer to a substation. Note that 0 is reserved for the do-nothing action.
        """

        self.num_to_sub = {i+1:k for i,k in enumerate(self.sub_id_to_action.keys())}
        self.num_to_sub[0] = 0

    def _get_tested_action(self, action_int:int) -> List[grid2op.Action.BaseAction]:
        """
        Returns the list of actions that will be tested by the simulator function.
        
        Parameters:
        ----------
        observation: :class:`grid2op.Observation.CompleteObservation`

        """

        self.sub_id_to_action[0] = [self.action_space({})] # list so it is compatible with greedy agent

        sub_id = self.num_to_sub[action_int]
        self.tested_action = self.sub_id_to_action[sub_id]

        return self.tested_action

    def act(self, observation: grid2op.Observation.CompleteObservation, sub_id:int) \
            -> grid2op.Action.BaseAction:
        """
        Simulates the actions for the given substations and chooses the one
        that brings down the max line capacity.

        Parameters:
        ----------
        observation: : grid2op.Observation.CompleteObservation
            The current observation of the :class:`grid2op.Environment.Environment`

        Returns
        -------
        res: grid2op.Action.Action
            The action that brings the rho the most.
        """
        self.tested_action = self._get_tested_action(sub_id)

        if len(self.tested_action) > 1: # do not iterate if only one action avaliable at a substation

            self.resulting_max_rho = np.full(shape=len(self.tested_action), fill_value=np.NaN, dtype=dt_float)
            for i, action in enumerate(self.tested_action):
                simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(action)

                # if (simul_obs.topo_vect == observation.topo_vect).all(): # if the topology is the same, do not compute the reward
                #     self.resulting_rewards[i] = float("-inf")
                # else:
                self.resulting_max_rho[i] = simul_obs.rho.max() if not simul_has_error else float("inf")

            min_rho_idx = int(np.argmin(self.resulting_max_rho))
            best_action = self.tested_action[min_rho_idx]

        else:
            best_action = self.tested_action[0]

        return best_action
