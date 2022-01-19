#from grid2op.Backend.PandaPowerBackend import PandaPowerBackend
import numpy as np
import grid2op
import os
import logging
import gym 
import time

from grid2op.PlotGrid import PlotMatplot 
from grid2op.gym_compat import GymEnv, MultiToTupleConverter, DiscreteActSpace,ScalerAttrConverter
from grid2op.Reward import L2RPNReward
from grid2op.Converter import IdToAct
from grid2op.gym_compat.gym_obs_space import GymObservationSpace
from grid2op.gym_compat.gym_act_space import GymActionSpace
from grid2op.gym_compat.utils import check_gym_version

from grid2op_env.medha_action_space import create_action_space, remove_redundant_actions
from grid2op_env.utils import CustomDiscreteActions
from grid2op_env.rewards import ScaledL2RPNReward
from lightsim2grid import LightSimBackend
from gym.spaces import Box # needed for adding connectivity matrix
from configurations import ROOT_DIR
 



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
                                        conn_matrix = env_config.get("conn_matrix", False))
        
        # Define parameters needed for parametric action space
        self.rho_threshold = env_config.get("rho_threshold", 0.95) - 1e-5 # used for stability in edge cases
        self.parametric_action_space = env_config["use_parametric"] and env_config["medha_actions"] and "rho" in env_config["keep_observations"]
        logging.info(f"Using parametric action space equals {self.parametric_action_space}")
        logging.info(f"The do nothing action is {self.do_nothing_actions}")

        self.run_until_threshold = env_config.get("run_until_threshold", False)
        self.reward_scaling_factor = env_config.get("reward_scaling_factor", 1) # useful for SAC
        self.log_reward = env_config.get("log_reward", False) # useful for SAC

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
            cum_reward = 0
            while (max(obs["rho"]) < self.rho_threshold) and (not done):
                cum_reward += reward
                obs, reward, done, info = self.env_gym.step(self.do_nothing_actions[0])
                self.steps += 1
            #reward = ((self.steps - self.begin_step)/100)*50 # experiment for sac
            reward = cum_reward*self.reward_scaling_factor 
            if self.log_reward:
                reward = np.log2(max(1,reward))
        return obs, reward, done, info

def create_gym_env(env_name = "rte_case14_realistic" , keep_obseravations = None, keep_actions = None, 
                    scale = True, convert_to_tuple = True, act_on_single_substation  = True,
                    medha_actions = True, seed=2137, disable_line = -1, conn_matrix = False, **kwargs):
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

    env = grid2op.make(env_name, reward_class = ScaledL2RPNReward, test = False, backend = LightSimBackend(), **kwargs)
    logging.info(f"Using {len(env.chronics_handler.subpaths)} chronics.")
    if seed is not None:
        logging.info(f"Setting the env seed to {seed}")
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        env.seed(seed)
    if medha_actions:
        logging.info("Using the action space and thermal limits defined by Medha!")
        if env_name != "rte_case14_realistic":
            raise NotImplementedError("Medha action space is only implemented for rte_case14_realistic")
        thermal_limits = [1000,1000,1000,1000,1000,1000,1000, 760,450, 760,380,380,760,380,760,380,380,380,2000,2000]
        env.set_thermal_limit(thermal_limits)

    # Convert to gym
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
                                                env_name,
                                                f"{attr}.npy"))#np.load(os.path.join(os.getcwd(), "/grid2op_env/scaling_arrays/", f"{attr}.npy"))
                env_gym.observation_space = env_gym.observation_space.\
                                            reencode_space(attr,
                                                ScalerAttrConverter(substract=c*min_arr,
                                                                    divide=c*(max_arr - min_arr)
                                                                    ))
    if conn_matrix:
        shape_ = (env.dim_topo, env.dim_topo)
        env_gym.observation_space.add_key("connectivity_matrix",
                                  lambda obs: obs.connectivity_matrix(),
                                  Box(shape=shape_,
                                      low=np.zeros(shape_),
                                      high=np.ones(shape_),
                                    )
                                  )
    if (act_on_single_substation) and (not medha_actions):

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


    return env_gym, do_nothing_actions, env, all_actions_dict

if __name__ == "__main__":
    logging.basicConfig(filename='env_create.log', filemode='w', level=logging.INFO)

    env = create_gym_env("rte_case14_realistic", keep_obseravations= ["rho", "gen_p"], convert_to_tuple=True)