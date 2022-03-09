import grid2op
import wandb
import operator

from typing import Tuple, List, Optional
from grid2op_env.grid_to_gym import create_gym_env

def get_do_nothing(env_config: Optional[dict] = None, wandb_run_id: Optional[str] = None) \
    -> Tuple[List[int], List[grid2op.Action.BaseAction]]  :
        """
        Returns the do nothing action index and the list of all actions.
        """
        # print("env config: ", env_config)
        # print("wandb run id: ", wandb_run_id)
        # print("(env_config is None),  (wandb_run_id is None): ", (env_config is None), (wandb_run_id is None), \
        #                                                 ((env_config is None) and (wandb_run_id is None)))
        assert operator.xor((env_config is None), (wandb_run_id is None)), \
            "env_config or wandb_key must be defined"

        if wandb_run_id:
            api = wandb.Api()
            # Fetch the environment config from one of the runs
            run = api.run(f"bmanczak/grid2op/{wandb_run_id}" )
            #run = api.run("bmanczak/grid2op/2e2dd_00001") # Non-greedy
            env_config = run.config["env_config"]

        _, do_nothing_actions, org_env, all_actions_dict = create_gym_env(env_name = env_config["env_name"],
                                        keep_obseravations= env_config["keep_observations"],
                                        keep_actions= env_config["keep_actions"],
                                        convert_to_tuple=env_config["convert_to_tuple"],
                                        act_on_single_substation=env_config["act_on_single_substation"],
                                        medha_actions=env_config["medha_actions"],
                                        scale = env_config.get("scale", False),
                                        disable_line = env_config.get("disable_line", -1),
                                        conn_matrix = env_config.get("conn_matrix", False),
                                        substation_actions = False,
                                        greedy_agent = False, # has to be false to return the do nothing actions properly
                                        graph_obs = env_config.get("graph_obs", False),
                                        combine_rewards= env_config.get("combine_rewards", False))
        
        return do_nothing_actions, all_actions_dict

def is_float(element) -> bool:
    try:
        return  float(element)
    except ValueError:
        return str(element)