import numpy as np
import math 
import argparse
import grid2op
import os
import logging
# import ray

from evaluation.restore_agent import restore_agent
from evaluation.rllib_to_grid import AgentFromGym, AgentThresholdEnv
from grid2op_env.grid_to_gym import Grid_Gym

logger = logging.getLogger(__name__)
logger.info("twoja matka")

def init_env(config: dict) ->  grid2op.Environment.Environment:
    '''
    Prepares the Grid2Op environment from a dictionary containing configuration
    setting.
    Parameters
    ----------
    config : dict
        Dictionary containing configuration variables.
    Returns
    -------
    env : TYPE
        The Grid2Op environment.
    '''
    data_path = config['paths']['rte_case14_realistic']
    scenario_path = config['paths']['rte_case14_realistic_chronics']

    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=data_path, chronics_path=scenario_path, backend=backend,
                           gamerules_class = grid2op.Rules.AlwaysLegal, test=True)
    except:
        env = grid2op.make(dataset=data_path, chronics_path=scenario_path,
                           gamerules_class = grid2op.Rules.AlwaysLegal, test=True)
        
    # for reproducible experiments
    env.seed(config['tutor_generated_data']['seed'])  

    #Set custom thermal limits
    thermal_limits = config['rte_case14_realistic']['thermal_limits']
    env.set_thermal_limit(thermal_limits)
    
    return env
    
def save_records(records: np.array, chronic: int, save_path: str, days_completed: int,
                 do_nothing_capacity_threshold: float, lout: int = -1,):
    '''
    Saves records of a chronic to disk and logger.infos a message that they are saved. 
    Parameters
    ----------
    records : np.array
        The records.
    chronic : int
        Int representing the chronic which is saved.
    save_path : str
        Path where the output folder with the records file is to be made.
    days_completed : int
        The number of days completed.
    do_nothing_capacity_threshold : int
        The threshold max. line rho at which the tutor takes actions.
    lout : int
        Index of any line that is out.
    '''

    
    folder_name = f'records_chronics_lout:{lout}_dnthreshold:{do_nothing_capacity_threshold}'
    file_name = f'records_chronic:{chronic}_dayscomp:{days_completed}.npy'
    if not os.path.isdir(os.path.join(save_path,folder_name)):
        os.mkdir(os.path.join(save_path,folder_name))
    np.save(os.path.join(save_path,folder_name,file_name), records)
    logger.info('# records are saved! #')
    
def empty_records(obs_vect_size: int):
    '''
    Generate a numpy array for storing records of actions and observations.
    Parameters
    ----------
    OBS_VECT_SIZE : int
        The size of the observation vector.
    Returns
    -------
    np.array
        The records numpy array.
    '''
    # first col for label, remaining OBS_VECT_SIZE cols for environment features
    return np.zeros((0, 5+obs_vect_size), dtype=np.float32)
    
def ts_to_day(ts: int):
    return math.floor(ts/ts_in_day)

def skip_to_next_day(env: grid2op.Environment.Environment,
                     num: int,
                     disable_line: int) -> dict:
    '''
    Skip the environment to the next day.
    Parameters
    ----------
    env : grid2op.Environment.Environment
        The environment to fast forward to the next day in.
    num : int
        The current chronic id.
    disable_line : int
        The index of the line to be disabled.
    Returns
    -------
    info : dict
        Grid2op dict given out as the fourth otuput of env.step(). Contains 
        the info about whether an error has occured.
    '''

    ts_next_day = ts_in_day*(1+ts_to_day(env.nb_time_step))
    env.set_id(num)
    _ = env.reset()
    
    if disable_line != -1:
        env.fast_forward_chronics(ts_next_day-1)
        _, _, _, info = env.step(env.action_space(
            {"set_line_status":(disable_line,-1) }))
        return info 
    else:
        #info = None
        env.fast_forward_chronics(ts_next_day)

    #return info

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_chronic_id",  help="The chronic to start with.",
                        required=False,default=0,type=int)
    parser.add_argument("--model_path",  help="The path to the model.", type = str,
                        required = False, default = "/Users/blazejmanczak/Desktop/SAC_Grid_Gym_86a3a_00001_1_disable_line=8,reward_scaling_factor=5_2021-12-23_13-26-27"  )
    parser.add_argument("--agent_type", help="The type of agent to use.", type = str,
                        choices=["sac", 'ppo'], required = False, default = "sac")
    parser.add_argument("--num_chronics", help="The number of chronics to use.", type = int,
                        required = False, default = 10)
    parser.add_argument("--checkpoint_num", default= None, type = int, help = "The checkpoint number to restore.")
    args = parser.parse_args()

    agent, env_config = restore_agent( 
                    path = args.model_path,
                   return_env_config = True, # disabled lines
                   trainer_type=args.agent_type,
                   checkpoint_num = args.checkpoint_num)

    rllib_env = Grid_Gym(env_config);

    num_chronics = args.num_chronics
    ts_in_day = 288 # 8064/28
    start_chronic_id = args.start_chronic_id #args.start_chronic_id
    do_nothing_capacity_threshold = env_config.get("rho_threshold",0.9)
    disable_line = env_config.get("disable_line", -1)

    print("-"*50, "\n")
    print(f"Evaluation of model with disabled line {disable_line} and model path {args.model_path} ")
    if env_config["use_parametric"]:
        wrapped_agent = AgentFromGym(rllib_env, agent)
    else:
        wrapped_agent = AgentThresholdEnv(rllib_env, agent, rho_threshold=do_nothing_capacity_threshold)

    #Initialize environment
    env = rllib_env.org_env
    env.reset()
    logger.info("Number of available scenarios: " + str(len(env.chronics_handler.subpaths)))
    env.set_id(start_chronic_id)

    #Prepare tutor and record objects

    obs_vect_size = len(env.get_obs().to_vect())
    records = empty_records(obs_vect_size)
    total_days = 0
    total_days_completed = 0

    for num in range(start_chronic_id, start_chronic_id+num_chronics):

        #(Re)set variables
        obs = env.reset()
        done,days_completed = False, 0
        day_records = empty_records(obs_vect_size)

        #Disable lines, if any
        if disable_line != -1:
            obs, _, _, info = env.step(env.action_space(
                            {"set_line_status":(disable_line,-1) }))
        else:
            info = {'exception':[]}

        logger.info('current chronic: %s' % env.chronics_handler.get_name())
        reference_topo_vect = obs.topo_vect.copy()

        while env.nb_time_step < env.chronics_handler.max_timestep():
            #Check for Diverging Powerflow exceptions, which happen sporadically
            if grid2op.Exceptions.PowerflowExceptions.DivergingPowerFlow in \
                        [type(e) for e in info['exception']]: 
                logger.info(f'Powerflow exception at step {env.nb_time_step} '+
                        f'on day {ts_to_day(env.nb_time_step)}')
                info = skip_to_next_day(env, num, disable_line)
                day_records = empty_records(obs_vect_size)
                continue
                
            try:
                obs = env.get_obs()
            except Exception as e:
                logger.warning(f'Exception {e} at step {env.nb_time_step}')
                print(f'Exception {e} at step {env.nb_time_step}')
                info = skip_to_next_day(env, num, disable_line)
                day_records = empty_records(obs_vect_size)

            
            #reset topology at midnight, store days' records, reset days' records
            if env.nb_time_step%ts_in_day == ts_in_day-1:
                logger.info(f'Day {ts_to_day(env.nb_time_step)} completed.')
                total_days_completed += 1
                total_days += 1 
                obs, _, _, _ = env.step(env.action_space({'set_bus': 
                                                        reference_topo_vect}))
                records = np.concatenate((records, day_records), axis=0)
                day_records = empty_records(obs_vect_size)
                days_completed += 1
                continue

            act = wrapped_agent.act(obs, reward = None, done= None)
            obs, reward, done, info = env.step(act)
                
            #If the game is done at this point, this indicated a (failed) game over
            #If so, reset the environment to the start of next day and discard the records
            if env.done:
                logger.info(f'Failure at step {env.nb_time_step - ts_to_day(env.nb_time_step)*ts_in_day} '+
                        f'on day {ts_to_day(env.nb_time_step)}')
                total_days += 1
                info = skip_to_next_day(env, num, disable_line)
                #day_records = empty_records(obs_vect_size)
        print(f"Percentage of days comleted {round(total_days_completed/total_days, 3)} after {num-start_chronic_id + 1} chronics.")
    print("Finised.")
        # logger.info whether game was completed succesfully, save days' records if so
    logger.info('Chronic exhausted! \n\n\n')
    #ray.shutdown()