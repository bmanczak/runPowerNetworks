from gym.spaces import Discrete

class CustomDiscreteActions(Discrete):
    """
    Class that customizes the action space.

    Example usage:
    
    import grid2op
    from grid2op.Converter import IdToAct
    
    env = grid2op.make("rte_case14_realistic")
    
    all_actions = # a list of of desired actions
    converter = IdToAct(env.action_space) 
    converter.init_converter(all_actions=all_actions) 


    env.action_space = ChooseDiscreteActions(converter=converter)


    """
    def __init__(self, converter):
        self.converter = converter
        Discrete.__init__(self, n=converter.n)
    def from_gym(self, gym_action):
        return self.converter.convert_act(gym_action)