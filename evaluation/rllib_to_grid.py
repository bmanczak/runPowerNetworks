from grid2op.Agent import BaseAgent 

class AgentFromGym(BaseAgent):
    def __init__(self, rllib_env, trained_agent):
        BaseAgent.__init__(self, rllib_env.action_space)
        self.trained_agent = trained_agent
        self.rllib_env = rllib_env
    def act(self, obs):
        
        gym_obs = self.rllib_env.obs_grid2rllib(obs)
        gym_act = self.trained_agent.compute_single_action(gym_obs)
    
        grid2op_act = self.rllib_env.action_rllib2grid(gym_act) #self.rllib_env.env_gym.action_space.from_gym(gym_act)

        return grid2op_act

class AgentThresholdEnv(BaseAgent):
    def __init__(self, rllib_env, trained_agent, rho_threshold = 0.9):
        BaseAgent.__init__(self, rllib_env.action_space)
        self.trained_agent = trained_agent
        self.rllib_env = rllib_env
        self.rho_threshold = rho_threshold

    def act(self, obs):
        
        if max(obs.rho) < self.rho_threshold:
            gym_act = 0 # do nothing 
        else:
            gym_obs = self.rllib_env.obs_grid2rllib(obs)
            gym_act = self.trained_agent.compute_single_action(gym_obs)

        grid2op_act = self.rllib_env.action_rllib2grid(gym_act) #self.rllib_env.env_gym.action_space.from_gym(gym_act)

        return grid2op_act

class AgentThresholdEnvGreedy(BaseAgent):
    def __init__(self, rllib_env, trained_agent, rho_threshold = 0.9):
        BaseAgent.__init__(self, rllib_env.action_space)
        self.trained_agent = trained_agent
        self.rllib_env = rllib_env
        self.steps = 0
        self.rho_threshold = rho_threshold

    def act(self, obs):
        
        gym_obs = self.rllib_env.obs_grid2rllib(obs)
        if max(obs.rho) < self.rho_threshold:
            gym_act = 0
        else:
            gym_act = self.trained_agent.compute_single_action(gym_obs)

        grid2op_act = self.rllib_env.env_gym.action_mapper(sub_id = gym_act, obs = obs) 
    
        return grid2op_act

class ClassicGreedyWrapper:

    def __init__(self, greedy_agent, rho_threshold = 0.95):
        self.agent = greedy_agent
        self.rho_threshold = rho_threshold

    def act(self, obs):
        if obs.rho.max() > self.rho_threshold:
                act = self.agent.act(obs)
                #obs, reward, done, info = self.env.step(act)
        else:
            act = self.agent.tested_action[self.agent.do_nothing_idx]
            #obs, reward, done, info = self.env.step(act)
        
        return act
