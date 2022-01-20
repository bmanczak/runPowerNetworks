from gym.spaces import Discrete
from collections import defaultdict, OrderedDict

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

def get_sub_id_to_elem_id(env):
    """
    Get the mapping between the substation id to the elements
    in the power grid.
    
    Keyword arguments:
    ----------
    env: Grid2Op_environment
        The environment.
    """
    
    elem_to_sub_id_arrays = [env.load_to_subid, env.gen_to_subid, env.line_or_to_subid, env.line_ex_to_subid, env.storage_to_subid]
    elem_to_elem_id = [env.reset().load_pos_topo_vect, env.reset().gen_pos_topo_vect, env.reset().line_or_pos_topo_vect, env.reset().line_ex_pos_topo_vect, env.reset().storage_pos_topo_vect]

    sub_id_to_elem_id = defaultdict(list) # necessary for pooling over substations

    for sub_id_arr, elem_id_arr in zip(elem_to_sub_id_arrays,elem_to_elem_id):
        for sub_id, elem_id in zip(sub_id_arr, elem_id_arr):
            sub_id_to_elem_id[sub_id].append(elem_id)

    sub_id_to_elem_id = OrderedDict(sorted(sub_id_to_elem_id.items()))

    return sub_id_to_elem_id

def get_sub_id_to_action(action_list):
    """
    Get the mapping between the substation id to the actions from 
    the action list.

    Assumes that each action in action_list only affects one substation.

    Args:
        action_list (list): List of Grid2Op actions.
    """

    sub_id_to_action = defaultdict(list)
    for action in action_list:
        for i, sub_id in enumerate(action.as_dict()["set_bus_vect"]["modif_subs_id"]):
            if i > 0:
                raise ValueError("Each action in action_list must only affects one substation.")
            else:
                sub_id_to_action[int(sub_id)].append(action)

    return sub_id_to_action


def reverse_dict(dic):
    """
    Reverses the dictionary with list values.
    """
    new_dic = {}
    for k,v in dic.items():
        v = [v] if type(v) is not list else v # for compatibility with non-list values
        for x in v:
            new_dic[x] = k
    return new_dic