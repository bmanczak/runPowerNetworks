from grid2op.Agent import BaseAgent 

class AgentFromGym(BaseAgent):
    def __init__(self, rllib_env, trained_agent):
        BaseAgent.__init__(self, rllib_env.action_space)
        self.trained_agent = trained_agent
        self.rllib_env = rllib_env
    def act(self, obs, reward, done):
        
        gym_obs = self.rllib_env.obs_grid2rllib(obs)
        gym_act = self.trained_agent.compute_action(gym_obs)
    
        grid2op_act = self.rllib_env.action_rllib2grid(gym_act) #self.rllib_env.env_gym.action_space.from_gym(gym_act)
        # if (gym_act != 0):
        #     print("Computed action: ", grid2op_act)
        return grid2op_act

class AgentThresholdEnv(BaseAgent):
    def __init__(self, rllib_env, trained_agent, rho_threshold = 0.9):
        BaseAgent.__init__(self, rllib_env.action_space)
        self.trained_agent = trained_agent
        self.rllib_env = rllib_env
        self.steps = 0
        self.rho_threshold = rho_threshold

    def act(self, obs, reward, done):
        
        if max(obs.rho) < 0.9:
            gym_act = 0 # do nothing 
        else:
            #print("taking action!", "rho", obs.rho)
            gym_obs = self.rllib_env.obs_grid2rllib(obs)
            gym_act = self.trained_agent.compute_action(gym_obs)
            self.steps += 1
            #print("number of steps: ", self.steps)
        #print("is done: ", done)
        # if done:
        #     print("Terminating having taken {} steps".format(self.steps))

        grid2op_act = self.rllib_env.action_rllib2grid(gym_act) #self.rllib_env.env_gym.action_space.from_gym(gym_act)
        # if (gym_act != 0):
        #     print("Computed action: ", grid2op_act)
        return grid2op_act