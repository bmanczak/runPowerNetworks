#from grid2op.Backend.PandaPowerBackend import PandaPowerBackend
import numpy as np
import grid2op
import os
import logging
import gym 
import random
import time

from collections import OrderedDict
from grid2op.PlotGrid import PlotMatplot 
from grid2op.gym_compat import GymEnv, MultiToTupleConverter, DiscreteActSpace,ScalerAttrConverter
from grid2op.Reward import L2RPNReward, CombinedReward
from grid2op.Converter import IdToAct
from grid2op.gym_compat.gym_obs_space import GymObservationSpace
from grid2op.gym_compat.gym_act_space import GymActionSpace
from grid2op.gym_compat.utils import check_gym_version
from grid2op.Agent.greedyAgent import GreedyAgent
from grid2op.dtypes import dt_float
from lightsim2grid import LightSimBackend

from gym.spaces import Box, Discrete # needed for adding connectivity matrix

from grid2op_env.medha_action_space import create_action_space, remove_redundant_actions
from grid2op_env.utils import CustomDiscreteActions, get_sub_id_to_action
from grid2op_env.rewards import ScaledL2RPNReward, CloseToOverflowReward, LinesReconnectedReward, DistanceReward
from grid2op_env.utils import get_sub_id_to_elem_id, reverse_dict, get_sub_id_to_action, opponent_kwargs
from models.utils import vectorize_obs
from models.greedy_agent import RoutingTopologyGreedy
from definitions import ROOT_DIR
from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import ModelCatalog

logger = logging.getLogger(__name__)
class CustomGymEnv(GymEnv):
    """
    fully implements the openAI gym API by using the :class:`GymActionSpace` and :class:`GymObservationSpace`
    for compliance with openAI gym.

    They can handle action_space_converter or observation_space converter to change the representation of data
    that will be fed to the agent.

    Notes
    ------
    The environment passed as input is copied. It is not modified by this "gym environment"

    Examples
    --------
    This can be used like:

    .. code-block:: python

        import grid2op
        from grid2op.gym_compat import GymEnv

        env_name = ...
        env = grid2op.make(env_name)
        gym_env = GymEnv(env)  # is a gym environment properly inheriting from gym.Env !
    """
    def __init__(self, env_init, disable_line = -1):
        super().__init__(env_init)
        self.disable_line = disable_line 
        self.reconnect_line = None
     
    def reset(self):
        if self.disable_line == -1:
            g2op_obs = self.init_env.reset()
        else:
            
            done = True
            i = -1
            while done:
                g2op_obs = self.init_env.reset()
                g2op_obs, _, done, info = self.init_env.step(self.init_env.action_space(
                                    {"set_line_status":(self.disable_line,-1) } ))
                i += 1
            if i!= 0:
                logging.info("Had to skip {} times to get a valid observation".format(i))
        
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs

    def step(self, gym_action):
        g2op_act = self.action_space.from_gym(gym_action)
        # print("g2op_act: ", g2op_act)
        # print_next_obs = False
        if self.reconnect_line is not None:
                # print("reconnecting!!!")
                reconnect_act = self.init_env.action_space(
                            {"set_line_status":(self.reconnect_line,1) })
                g2op_act = g2op_act + reconnect_act
                print_next_obs = True
                self.reconnect_line = None
                
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        # if info['is_illegal'] or info["is_ambiguous"]:
        #     print("illegal action: ", g2op_act)
        #     print("Info",info)
        # if (g2op_obs.topo_vect == -1).any():
        #     print("g2op_obs rho: ", g2op_obs.rho)
        # if print_next_obs:
        #     print("next obs: ", g2op_obs.rho)

        if isinstance(info["opponent_attack_line"], np.ndarray):
                if info["opponent_attack_duration"] == 1:
                    line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()[0]   
                    self.reconnect_line = line_id_attacked  

        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, float(reward), done, info

class SubstationGreedyEnv(CustomGymEnv):

    def __init__(self,env_init, greedy_agent, disable_line = -1, graph_obs = False):
        super().__init__(env_init, disable_line)
        self.graph_obs = graph_obs
        self.agent = greedy_agent
        self.reconnect_line = None
    
    def action_mapper(self, sub_id, obs = None):
        """
        Map the action to the substation id.
        """
        if obs is None:
            obs = self.last_obs 
        return self.agent.act(obs, sub_id)

    def reset(self):
        if self.disable_line == -1:
            g2op_obs = self.init_env.reset()
        else:
            done = True
            i = -1
            while done:
                g2op_obs = self.init_env.reset()
                g2op_obs, _, done, info = self.init_env.step(self.init_env.action_space(
                                    {"set_line_status":(self.disable_line,-1) } ))
                i += 1
            if i!= 0:
                logging.info("Had to skip {} times to get a valid observation".format(i))
        self.last_obs = g2op_obs # needed for the greedy agent
        # if self.graph_obs:
        #     print("Vectorizing the observation!!")
        #     gym_obs = self.observation_space.to_gym(g2op_obs)
        #     gym_obs = vectorize_obs(gym_obs, self.init_env)
        #     print("Shape of vectorized", gym_obs.shape)
        # else:
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs

    def step(self, gym_action):
        g2op_act = self.action_mapper(sub_id = gym_action)
        # print_next_obs = False
        if self.reconnect_line is not None:
                # print("reconnecting!!!")
                reconnect_act = self.init_env.action_space(
                            {"set_line_status":(self.reconnect_line,1) })
                g2op_act = g2op_act + reconnect_act
                print_next_obs = True
                self.reconnect_line = None

        # print("gym action", gym_action)
        self.g2op_act = g2op_act
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        self.last_obs = g2op_obs # needed for the greedy agent
        # if (g2op_obs.topo_vect == -1).any():
        #     print("g2op_obs rho: ", g2op_obs.rho)
        # if print_next_obs:
        #     print("next obs: ", g2op_obs.rho)

        if isinstance(info["opponent_attack_line"], np.ndarray):
                if info["opponent_attack_duration"] == 1:
                    line_id_attacked = np.argwhere(info["opponent_attack_line"]).flatten()[0]   
                    self.reconnect_line = line_id_attacked  
        # if self.graph_obs:
        #     print("Vectorizing the observation!!")
        #     gym_obs = self.observation_space.to_gym(g2op_obs)
        #     gym_obs = vectorize_obs(gym_obs, self.init_env)
        #     print("Shape of vectorized", gym_obs.shape)
        # else:
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, float(reward), done, info
    
    def close(self):
        """
        Removes the .close() method from GymActionSpace.
        """
        if hasattr(self, "init_env") and self.init_env is None:
            self.init_env.close()
            del self.init_env
        self.init_env = None
        if hasattr(self, "observation_space") and self.observation_space is not None:
            self.observation_space.close()
        self.observation_space = None

class Grid_Gym(gym.Env):
    """
    A wrapper for the gym env required by RLLib.
    """
    def __init__(self, env_config):
        
        self.env_gym, self.do_nothing_actions, self.org_env, self.all_actions_dict = create_gym_env(env_name = env_config["env_name"],
                                        keep_obseravations= env_config["keep_observations"],
                                        keep_actions= env_config["keep_actions"],
                                        convert_to_tuple=env_config["convert_to_tuple"],
                                        act_on_single_substation=env_config["act_on_single_substation"],
                                        medha_actions=env_config["medha_actions"],
                                        scale = env_config.get("scale", False),
                                        disable_line = env_config.get("disable_line", -1),
                                        conn_matrix = env_config.get("conn_matrix", False),
                                        substation_actions = env_config.get("substation_actions", False),
                                        greedy_agent = env_config.get("greedy_agent", False),
                                        graph_obs = env_config.get("graph_obs", False),
                                        combine_rewards= env_config.get("combine_rewards", False),
                                        with_opponent= env_config.get("with_opponent", False))
        
        # Define parameters needed for parametric action space
        self.rho_threshold = env_config.get("rho_threshold", 0.95) - 1e-5 # used for stability in edge cases
        self.parametric_action_space = env_config["use_parametric"] and env_config["medha_actions"] and "rho" in env_config["keep_observations"]
        logging.info(f"Using parametric action space equals {self.parametric_action_space}")
        logging.info(f"The do nothing action is {self.do_nothing_actions}")

        self.run_until_threshold = env_config.get("run_until_threshold", False)
        self.reward_scaling_factor = env_config.get("reward_scaling_factor", 1) # useful for SAC
        self.log_reward = env_config.get("log_reward", False) # useful for SAC
        self.steps = 0 # useful for tracking number of steps in the real environment

        self.disable_line = env_config.get("disable_line", -1)
        if self.run_until_threshold and self.parametric_action_space:
            logging.warning("run_until_threshold is not compatible with parametric action space. Setting run_until_threshold to False")
            self.run_until_threshold = False

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


    def update_avaliable_actions(self, mask_topo_change):
        """
        Masks the actions that change the topology of the grid.

        Args:
            mask_topo_change (bool): if True, only the do-nothing actions are not masked.
        """
        if mask_topo_change:
            self.action_mask = np.array(
                                            [0.] * self.env_gym.action_space.n, dtype=np.float32)
            self.action_mask[self.do_nothing_actions] = 1. # hack: the 0-th action is doing nothing for all configurations.
        else:
            self.action_mask = np.array(
                                        [1.] * self.env_gym.action_space.n, dtype=np.float32)
       

    def obs_grid2rllib(self, g2op_obs):
        """
        Transforms grid2op obs to gym obs.
        """
        obs = self.env_gym.observation_space.to_gym(g2op_obs)
        obs = OrderedDict((k, obs[k]) for k in self.observation_space.spaces)
        if self.parametric_action_space:
            mask_topo_change = max(obs["rho"]) < self.rho_threshold
            self.update_avaliable_actions(mask_topo_change)
            return {"action_mask": self.action_mask, "grid": obs}
        else:
            return obs
    
    def action_rllib2grid(self, action_rllib):
        """
        Transform the action from rllib to grid2op.
        """
        return self.env_gym.action_space.from_gym(action_rllib)

    def reset(self):
        obs = self.env_gym.reset()
        if self.parametric_action_space:
            mask_topo_change = max(obs["rho"]) < self.rho_threshold
            self.update_avaliable_actions(mask_topo_change)
            # if not mask_topo_change:
            #     print("all actions avaliable", max(obs["rho"]))
            #     print("sum of the mask", self.action_mask)

            return {"action_mask": self.action_mask, "grid": obs}

        elif self.run_until_threshold:
            done = False
            self.steps = 0
            # print("Entering the loop")
            #start = time.time()
            while (max(obs["rho"]) < self.rho_threshold) and (not done):
                obs, _, done, _ = self.env_gym.step(self.do_nothing_actions[0])
                self.steps += 1
        return obs


    def step(self, action):
       
        obs, reward, done, info = self.env_gym.step(action)
        if self.parametric_action_space:
            mask_topo_change = max(obs["rho"]) < self.rho_threshold
            self.update_avaliable_actions(mask_topo_change)
            obs = {"action_mask": self.action_mask, "grid": obs}

        elif self.run_until_threshold:
            self.begin_step = self.steps
            cum_reward = reward
            while (max(obs["rho"]) < self.rho_threshold) and (not done):
                #cum_reward += reward
                obs, reward, done, info = self.env_gym.step(self.do_nothing_actions[0])
                cum_reward += reward
                self.steps += 1
            #reward = ((self.steps - self.begin_step)/100)*50 # experiment for sac
            # do_nothing_steps = max(1, self.steps - self.begin_step)
            
            # if random.uniform(0,1) < 0.02: # 2% of the time print
            #     print("action reward is", reward)
                
            #     scaled_no_op = (np.log2(max(1,do_nothing_steps))/do_nothing_steps)*(cum_reward)
            #     print("scaled reward is", scaled_no_op)
            #     if scaled_no_op > 0:
            #         print("The ratio is", reward / scaled_no_op )

            # reward = reward + (1/do_nothing_steps)*cum_reward # 
            reward = cum_reward*self.reward_scaling_factor 
            if done:
                info["steps"] = self.steps
            if self.log_reward:
                reward = np.log2(max(1,reward))
        return obs, reward, done, info


class HierarchicalGridGym(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()

        self.action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(106))
        self._skip_env_checking = True
        
        self.env_gym = Grid_Gym(env_config)
        self.org_env = self.env_gym.org_env
        
        self.low_level_agent_id = "choose_action_agent"
        self.high_level_agent_id = "choose_substation_agent"

        self.sub_id_to_action_num = get_sub_id_to_action(self.env_gym.all_actions_dict,
                return_action_ix= True)
        self.num_to_sub = {i:k for i,k in enumerate(self.sub_id_to_action_num.keys())}
        self.info = {"steps": 0}

        logger.debug("The sub_id_to_action_num is", self.sub_id_to_action_num)
        logger.debug("The num_to_sub is", self.num_to_sub)

    def map_sub_to_mask(self):
        """
        Produces a mask given for the low level agent 
        given the prediction of the high level agent.
        """
        
        action_mask = np.array([0.] * 106, dtype=np.float32)
        modified_sub = self.num_to_sub[self.high_level_pred]  
        aval_actions = self.sub_id_to_action_num[modified_sub]
        action_mask[aval_actions] = 1.

        return action_mask

    def reset(self):
        self.cur_obs = self.env_gym.reset()
        self.high_level_pred = None # the substation to modify
        self.steps_remaining_at_level = None
    
        one_hot_encoded_action = np.zeros(106)

        obs = {self.high_level_agent_id: {
                        "regular_obs": self.cur_obs,
                        "chosen_action": 0}
        }
        return obs

    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if self.high_level_agent_id in action_dict:
            return self._high_level_step(action_dict["choose_substation_agent"])
        else:
            return self._low_level_step(list(action_dict.values())[0])

    def _high_level_step(self, action, cur_obs = None):
        logger.debug("High level agent sets goal")
        self.high_level_pred = action       
        # Create a mask using the predicited action
        action_mask = self.map_sub_to_mask()
        
        if cur_obs is not None:
            self.cur_obs = cur_obs
            
        obs = {self.low_level_agent_id: {
            "action_mask": action_mask,
            "regular_obs":self.cur_obs ,
            "chosen_substation": self.high_level_pred,
        }}
        rew = {self.low_level_agent_id: 0}
        done = {"__all__": False}
        return obs, rew, done, {self.low_level_agent_id: self.info}

    def _low_level_step(self, action):
        logger.debug("Low level agent step {}".format(action))
        # Step in the actual env
        f_obs, f_rew, f_done, f_info = self.env_gym.step(action)
        # Get the number of survived steps
        self.info["steps"] = f_info.get("steps", 0)
        self.cur_obs = f_obs

        # Calculate low-level agent observation and reward
        rew = {self.low_level_agent_id: f_rew}

        # Handle env termination & transitions back to higher level
        done = {"__all__": False}
        if f_done:
            done["__all__"] = True
            logger.debug("high level final reward {}".format(f_rew))

        one_hot_encoded_action = np.zeros(106)
        one_hot_encoded_action[action] = 1
        rew = {self.low_level_agent_id: f_rew,
               self.high_level_agent_id: f_rew}

        obs = {self.high_level_agent_id: {
                        "regular_obs": f_obs,
                        "chosen_action": action}
        }

        return obs, rew, done, {self.high_level_agent_id: self.info}

class Grid_Gym_Greedy(Grid_Gym):

    def __init__(self, env_config):
        super().__init__(env_config)
        
        self.action_space = gym.spaces.Discrete(self.env_gym.action_space.n) # then already discrete
        self.steps = 0
        #self.graph_obs = env_config.get("graph_obs", False)
        # self.action_space = gym.spaces.Discrete(self.env_gym.action_space.n) 
        # self.observation_space =self.env_gym.observation_space
        print("I am here!!!!")
        print("Dim action space", self.env_gym.action_space.n)
        print("Observation space", self.env_gym.observation_space)
        logging.info(f"Dim action space: {self.env_gym.action_space.n}")
    
    def reset(self):
        print(f"Survived {self.steps} real environment steps!")
        obs = self.env_gym.reset()
        
        done = False
        self.steps = 0
        while (max(obs["rho"]) < self.rho_threshold) and (not done):
            obs, _, done, _ = self.env_gym.step(0)
            self.steps += 1
         # See https://discuss.ray.io/t/preprocessor-fails-on-observation-vector/614
        # order matters
        obs = OrderedDict((k, obs[k]) for k in self.observation_space.spaces)
        return obs

    def step(self, action):
       
        obs, reward, done, info = self.env_gym.step(action)
        self.begin_step = self.steps
        cum_reward = reward
        while (max(obs["rho"]) < self.rho_threshold) and (not done):
           # cum_reward += reward
            obs, reward, done, info = self.env_gym.step(0)
            cum_reward += reward
            self.steps += 1
        reward = cum_reward*self.reward_scaling_factor 
        if done:
            info["steps"] = self.steps
        if self.log_reward:
            reward = np.log2(max(1,reward))
        # See https://discuss.ray.io/t/preprocessor-fails-on-observation-vector/614
        # order matters
        obs = OrderedDict((k, obs[k]) for k in self.observation_space.spaces)
        return obs, reward, done, info 
        
def create_gym_env(env_name = "rte_case14_realistic" , keep_obseravations = None, keep_actions = None, 
                    scale = True, convert_to_tuple = True, act_on_single_substation  = True,
                    medha_actions = True, seed=2137, disable_line = -1, conn_matrix = False,
                    substation_actions = False, greedy_agent = False, graph_obs = False, combine_rewards =False,
                    with_opponent = False,
                    **kwargs):
    """
    Create a gym environment from a grid2op environment.

    Keyword arguments:
    ----------
    env_name: str
        Name of the grid2op environment to create.
    keep_observations: list(str)
        List of observation to keep. If None, all observation are kept.
    keep_actions: list(str)
        List of action to keep. If None, all action are kept.
        Ignored if medha_actions is True.
    scale: bool
        Not implemented yet.
    convert_to_tuple: bool
        If True, the MultiBinary actions are converted to tuple.
        Ensures compatibility with rlib. Ignored if medha_actions or 
        act_on_single_substation is True.
    medha_actions: bool
        Whether to use a custom action space as specifed by Medha.
        If True, act_on_single_substation, keep_actions and act_on_single_substation are ignored. 
    seed: int
        Seed used to initialize the environment.
    disable_line: int
        If not -1, the line with the given id is disabled.
    conn_matrix: bool
        If True, the connectivity matrix will be added to each observation.
    substation_actions: bool
        If True, the action space will be discrete, with n equal to the number of actionable substations.
    greedy_agent: bool
        If True, the greedy agent will be used to act once the substations is chosen.
    **kwargs:
        All the parameters of the grid2op environment.
        Most imporant parameters are:
        - reward_class: int

    Returns:
    -------
    env_gym: GymEnv
        The gym environment.
    do_nothing_actions: list(int)
        List of the do-nothing actions.
    env: grid2op env
        The original grid2op environment.
    """
    
    if combine_rewards:
        if with_opponent:
            env = grid2op.make(env_name, reward_class = CombinedReward, test = False, backend = LightSimBackend(), **opponent_kwargs)
        else: 
            env = grid2op.make(env_name, reward_class = CombinedReward, test = False, backend = LightSimBackend(), **kwargs)
        cr = env.get_reward_instance()
        cr.addReward("LinesReconnectedReward", LinesReconnectedReward(), 0.333)
        cr.addReward("CloseToOverflowReward", CloseToOverflowReward(), 0.333)
        cr.addReward("ScaledL2RPNReward", ScaledL2RPNReward(), 0.333)
        #cr.addReward("DistanceReward", DistanceReward(), 1.0)
        cr.initialize(env)
    else:
        print("ENV NAME", env_name)
        if with_opponent:
            env = grid2op.make(env_name, reward_class = ScaledL2RPNReward, test = False, backend = LightSimBackend(), **opponent_kwargs)
        else:
            env = grid2op.make(env_name, reward_class = ScaledL2RPNReward, test = False, backend = LightSimBackend(), **kwargs)
    
    print(f"The environment has {len(env.chronics_handler.subpaths)} chronics.")
    logging.info(f"The reward range is {env.reward_range}")
    logging.info(f"Using {len(env.chronics_handler.subpaths)} chronics.")
    if seed is not None:
        logging.info(f"Setting the env seed to {seed}")
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        env.seed(seed)
    if medha_actions:
        logging.info("Using the action space and thermal limits defined by Medha!")
        if "rte_case14_realistic" not in env_name:
            raise NotImplementedError("Medha action space is only implemented for rte_case14_realistic")
        thermal_limits = [1000,1000,1000,1000,1000,1000,1000, 760,450, 760,380,380,760,380,760,380,380,380,2000,2000]
        env.set_thermal_limit(thermal_limits)

    # Convert to gym
     # Convert to gym
    if greedy_agent:
        print("Using greedy agent")
        agent = RoutingTopologyGreedy(env.action_space, {}) # init a greedy agent with an empy mapping
        env_gym = SubstationGreedyEnv(env, agent , disable_line=disable_line, graph_obs = graph_obs)
    else:
        env_gym = CustomGymEnv(env, disable_line=disable_line)
    logging.info("Environment successfully converted to Gym")

    if keep_obseravations is not None:
        env_gym.observation_space = env_gym.observation_space.keep_only_attr(keep_obseravations)
    
    if scale:
        env_gym.observation_space = env_gym.observation_space.\
                                    reencode_space("gen_p",
                                           ScalerAttrConverter(substract=0.,
                                                               divide=env.gen_pmax
                                                               ))
        env_gym.observation_space = env_gym.observation_space.\
                                    reencode_space("timestep_overflow",
                                   ScalerAttrConverter(substract=0.,
                                                       divide=grid2op.Parameters.Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED # assuming no custom params
                                    ))

        for attr in ["p_ex", "p_or", "load_p"]:
            if keep_obseravations is None or attr in keep_obseravations:
                c = 1.2 # constant to account that our max/min are underestimated
                max_arr, min_arr = np.load(os.path.join(ROOT_DIR,
                                                "grid2op_env/scaling_arrays",
                                                "rte_case14_realistic",
                                                f"{attr}.npy"))#np.load(os.path.join(os.getcwd(), "/grid2op_env/scaling_arrays/", f"{attr}.npy"))
                env_gym.observation_space = env_gym.observation_space.\
                                            reencode_space(attr,
                                                ScalerAttrConverter(substract=c*min_arr,
                                                                    divide=c*(max_arr - min_arr)
                                                                    ))
    if conn_matrix:
        shape_ = (env.dim_topo, env.dim_topo)
        static_connectivity_matrix = env.reset().connectivity_matrix(as_csr_matrix=False) # all elements on bar 1
        env_gym.observation_space.add_key("connectivity_matrix", # this matrix shows what elements can theoretically be connected to each other
                                  lambda obs: static_connectivity_matrix, #obs.connectivity_matrix()
                                  Box(shape=shape_,
                                      low=np.zeros(shape_),
                                      high=np.ones(shape_),
                                    )
                                  )           
    if (act_on_single_substation) and (not medha_actions) and (not substation_actions):

        if keep_actions is not None:
            env_gym.action_space =DiscreteActSpace(env.action_space,
                                    attr_to_keep  = keep_actions )
        else:
            logging.info("All actions from the {env_name} environment are kept.")
            env_gym.action_space = DiscreteActSpace(env.action_space)
    
    elif not act_on_single_substation: # actions allowed on multiple substations
        logging.warning("Actions on multiple substations are allowed!")

        if keep_actions is not None:
            env_gym.action_space = env_gym.action_space.keep_only_attr(keep_actions)
    
        if convert_to_tuple: # DiscreteActSpace does not require conversion
            for action_type in env_gym.action_space:
                if type(env_gym.action_space[action_type]) == gym.spaces.multi_binary.MultiBinary:
                    env_gym.action_space = env_gym.action_space.reencode_space(action_type, MultiToTupleConverter())
                    logging.info(f"Converted action {action_type} to tuple")
        
    if medha_actions: # add action space from medha

        all_actions_with_redundant, reference_substation_indices, all_actions_dict_with_redundant = create_action_space(
                                                                                                    env,
                                                                                                    disable_line = disable_line,
                                                                                                    return_actions_dict=True)  # used in the Grid_Gym converter to only get the data above the threshold
        all_actions, do_nothing_actions, all_actions_dict = remove_redundant_actions(all_actions_with_redundant, reference_substation_indices,
                                                                nb_elements=env.sub_info, all_actions_dict= all_actions_dict_with_redundant)

        converter = IdToAct(env.action_space)  # initialize with regular the environment of the regular action space
        converter.init_converter(all_actions=all_actions) 

        env_gym.action_space = CustomDiscreteActions(converter = converter)

    if substation_actions:
        print("Entered substation actions")
        
        all_actions_with_redundant, reference_substation_indices, all_actions_dict_with_redundant = create_action_space(
                                                                                                    env,
                                                                                                    disable_line = disable_line,
                                                                                                    return_actions_dict=True)  # used in the Grid_Gym converter to only get the data above the threshold
        all_actions, do_nothing_actions, _ = remove_redundant_actions(all_actions_with_redundant, reference_substation_indices,
                                                                nb_elements=env.sub_info, all_actions_dict= all_actions_dict_with_redundant,
                                                                remove_all_redundant_actions=True)

        env_gym.agent.sub_id_to_action = get_sub_id_to_action(all_actions)
        env_gym.agent.get_num_to_sub()

        num_actionable_subs = len(env_gym.agent.sub_id_to_action.keys())
        print("Num actionable subs", num_actionable_subs)
        env_gym.action_space = Discrete(num_actionable_subs + 1) # +1 for the do nothing action


    return env_gym, do_nothing_actions, env, all_actions


def get_env_spec(env_config:dict):
    """
    Get the constants of the environment.
    
    Keyword arguments:
    ----------
    env_config: dict
        The dictionary with the environment configuration.
    """
    env_gym, _, env, all_actions = create_gym_env(env_name = env_config["env_name"],
                                        keep_obseravations= env_config["keep_observations"],
                                        keep_actions= env_config["keep_actions"],
                                        convert_to_tuple=env_config["convert_to_tuple"],
                                        act_on_single_substation=env_config["act_on_single_substation"],
                                        medha_actions=env_config["medha_actions"],
                                        scale = env_config.get("scale", False),
                                        disable_line = env_config.get("disable_line", -1),
                                        conn_matrix = env_config.get("conn_matrix", False),
                                        substation_actions = env_config.get("substation_actions", False),
                                        greedy_agent = env_config.get("greedy_agent", False),
                                        graph_obs = env_config.get("graph_obs", False))
    sub_id_to_elem_id = {k:sorted(v) for k,v in get_sub_id_to_elem_id(env).items()}
    topo_spec = env.action_space # holds the topology of the elements
    
    sub_id_to_action = get_sub_id_to_action(all_actions)

    line_to_sub_id = (env.reset().line_or_to_subid, env.reset().line_ex_to_subid)

    return sub_id_to_elem_id, topo_spec, sub_id_to_action, line_to_sub_id



if __name__ == "__main__":
    logging.basicConfig(filename='env_create.log', filemode='w', level=logging.INFO)

    env = create_gym_env("rte_case14_realistic", keep_obseravations= ["rho", "gen_p"], convert_to_tuple=True)