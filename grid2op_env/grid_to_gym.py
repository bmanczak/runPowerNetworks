#from grid2op.Backend.PandaPowerBackend import PandaPowerBackend
import numpy as np
import grid2op
import os
import logging
import gym 

from grid2op.PlotGrid import PlotMatplot 
from grid2op.gym_compat import GymEnv, MultiToTupleConverter, DiscreteActSpace,ScalerAttrConverter
from grid2op.Reward import L2RPNReward
from grid2op.Converter import IdToAct
from torch.functional import meshgrid

from grid2op_env.medha_action_space import create_action_space, remove_redundant_actions
from grid2op_env.utils import CustomDiscreteActions
from lightsim2grid import LightSimBackend


class Grid_Gym(gym.Env):
    """
    A wrapper for the gym env required by RLLib.
    """
    def __init__(self, env_config):
        
        self.env_gym, self.do_nothing_actions, self.org_env = create_gym_env(env_name = env_config["env_name"],
                                        keep_obseravations= env_config["keep_observations"],
                                        keep_actions= env_config["keep_actions"],
                                        convert_to_tuple=env_config["convert_to_tuple"],
                                        act_on_single_substation=env_config["act_on_single_substation"],
                                        medha_actions=env_config["medha_actions"],
                                        scale = env_config.get("scale", False))
        
        # Define parameters needed for parametric action space
        self.rho_threshold = env_config.get("rho_threshold", 0.95) - 1e-5 # used for stability in edge cases
        self.parametric_action_space = env_config["use_parametric"] and env_config["medha_actions"] and "rho" in env_config["keep_observations"]
        logging.info(f"Using parametric action space equals {self.parametric_action_space}")
        logging.info(f"The do nothing action is {self.do_nothing_actions}")

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
        return obs


    def step(self, action):
       
        obs, reward, done, info = self.env_gym.step(action)
        # if (action != 0) and (not done):
        #     print("action", action)
        #     print("info", info)
        #     print("topo vect", obs["topo_vect"])
        if self.parametric_action_space:
            mask_topo_change = max(obs["rho"]) < self.rho_threshold
            self.update_avaliable_actions(mask_topo_change)
            obs = {"action_mask": self.action_mask, "grid": obs}

        return obs, reward, done, info

def create_gym_env(env_name = "rte_case14_realistic" , keep_obseravations = None, keep_actions = None, 
                    scale = True, convert_to_tuple = True, act_on_single_substation  = True,
                    medha_actions = True, seed=2137, **kwargs):
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

    env = grid2op.make(env_name, reward_class = L2RPNReward, test = False, backend = LightSimBackend(), **kwargs)
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
    env_gym = GymEnv(env)
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
                max_arr, min_arr = np.load(os.path.join(os.getcwd(),
                                                "grid2op_env/scaling_arrays",
                                                env_name,
                                                f"{attr}.npy"))#np.load(os.path.join(os.getcwd(), "/grid2op_env/scaling_arrays/", f"{attr}.npy"))
                env_gym.observation_space = env_gym.observation_space.\
                                            reencode_space(attr,
                                                ScalerAttrConverter(substract=c*max_arr,
                                                                    divide=c*(max_arr - min_arr)
                                                                    ))

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

        all_actions_with_redundant, reference_substation_indices = create_action_space(env)  # used in the Grid_Gym converter to only get the data above the threshold
        all_actions, do_nothing_actions = remove_redundant_actions(all_actions_with_redundant, reference_substation_indices,
                                                                nb_elements=env.sub_info)

        converter = IdToAct(env.action_space)  # initialize with regular the environment of the regular action space
        converter.init_converter(all_actions=all_actions) 

        env_gym.action_space = CustomDiscreteActions(converter = converter)


    return env_gym, do_nothing_actions, env

if __name__ == "__main__":
    logging.basicConfig(filename='env_create.log', filemode='w', level=logging.INFO)

    env = create_gym_env("rte_case14_realistic", keep_obseravations= ["rho", "gen_p"], convert_to_tuple=True)