# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:49:51 2020

@author: Medha and Jan

This module contains functions to create a list of all possible actions.

Each action corresponds to a specific configuration of a specific substation.
An action is given by a dictionary {"set_bus" : substation_config} where
substation_config is again a dictionary of the form

substation_config = {"loads_id" : [(id1,busbar_nr), (id2,busbar_nr), ...], 
                     "generators_id" : [(id1,busbar_nr), (id2,busbar_nr), ...],
                     "lines_or_id" : [(id1,busbar_nr), (id2,busbar_nr), ...],
                     "lines_ex_id" : [(id1,busbar_nr), (id2,busbar_nr), ...]}

which specifies how the included assets (given via their ids) are connected to
busbars. For example,

substation_config = {'loads_id': [], 
                     'generators_id': [(4, 1)],
                     'lines_or_id': [(0, 1), (1, 1)],
                     'lines_ex_id': []}
This substation_config is represented by the variable "act_dict"

Note that for the actions space the ids have to be chosen consistently,
that is, related to only one substation.

The list of actions is created by looping through all substations and creating
all valid substation configurations for each substation.
A crucial command to do this is:
env.get_obj_connect_to(None, substation_nr).items()

which gives the ids of all the elements connected to a specific substation as
well as the total number of elements connected. For example, for 
substation_nr=1 one obtains

dict_items([('loads_id', array([0])),
            ('generators_id', array([0])),
            ('lines_or_id', array([2, 3, 4])),
            ('lines_ex_id', array([0])), 
            ('nb_elements', 6)])

The actions are constructed in such a way that they satisfy the substation
related constraints. However, the connex constraint is not checked.

After substation_config is created (or "act_dict"), this needs to be translated 
into a format to be used within the grid2op package which defines how an action
is displayed. It is of a datatype called "TopologyAction" which
is derived from "Action_space" method and it needs a dictionary of the type:
    {"set_bus":substation_config} 
    where substation is the "key" of this dictionary
Another key that this "Action_space" method can take is "change_bus" but that 
is not considered here. 
"""
import grid2op
import numpy as np
import itertools as it

from typing import List, Tuple

def create_dictionary(combs,sub_elem): 
    """ To create action in the form of dictionary for this particular 
    input combination for this list of elements
    #the input combination provided pertains to the combination for a single 
    # busbar, i.e. busbar number 1. 
    # The combination for busbar number 2 is extracted from here in the manner
    that if the input combination (combs) does not contain elements which are
    not in the list of elements (sub_elem) then these elements will form the
    combination for busbar 2."""
    sub_elem_local=sub_elem.copy() #to create a local copy of sub_elem
    #variables to count the number of loads, gens and lines in this combination
    ctr_load=0
    ctr_gen=0
    ctr_line_or=0
    ctr_line_ex=0
    elem1=[]
    elem2=[]
    elem3=[]
    elem4=[]
    elem5=[]
    elem6=[]
    elem7=[]
    elem8=[]
    copy_combs=sub_elem_local.copy()
    for elem in sub_elem_local:  
        #check if elements in sub_elem_local and then remove them
        if elem in combs: 
        # removing elements from overall list since they will be
        # assigned to busbar1
            copy_combs.remove(elem)
        # copy_combs will be the combination of elements for busbar 2
    # this below loop will create the required setup for bus bar 1
    for elem in combs: 
        #to create the dictionary related to elements on busbar 1
        if 'loads_id' in elem:
            length=len(elem)-8 # 8 is length of string 'loads_id', 
            # done in this manner for 2 or more digit ids
            elem1.append(int(elem[-length:])) 
            ctr_load+=1
        elif 'generators_id' in elem:
            length=len(elem)-13  # 13 is length of string 'generators_id'
            elem2.append(int(elem[-length:]))
            ctr_gen+=1
        elif 'lines_or_id' in elem:
            length=len(elem)-11 # 11 is length of string 'lines_or_id'
            elem3.append(int(elem[-length:]))
            ctr_line_or+=1      
        elif 'lines_ex_id' in elem:
            length=len(elem)-11  # 11 is length of string 'lines_ex_id'
            elem4.append(int(elem[-length:]))
            ctr_line_ex+=1
        
    # to create the list of '1's in order to create the pair as required for 
    # the four types of elements: loads, gens, lines_or, lines_ex
    busbar_loads=[1]*ctr_load 
    #loads=list(zip(elem1,busbar1))
    busbar_gens=[1]*ctr_gen
    #gens=list(zip(elem2,busbar1))
    busbar_lines_or=[1]*ctr_line_or
    #lines_or=list(zip(elem3,busbar1))
    busbar_lines_ex=[1]*ctr_line_ex
    #lines_ex=list(zip(elem4,busbar1))
    
    
    # resetting the variables
    ctr_load=0
    ctr_gen=0
    ctr_line_ex=0
    ctr_line_or=0
    # this below loop will create the required setup for bus bar 2
    for elem in copy_combs: 
        if 'loads_id' in elem:
            length=len(elem)-8 # 8 is length of string 'loads_id'
            # done in this manner for 2 or more digit ids
            elem5.append(int(elem[-length:])) 
            ctr_load+=1
        elif 'generators_id' in elem:
            length=len(elem)-13  # 13 is length of string 'generators_id'
            elem6.append(int(elem[-length:]))
            ctr_gen+=1
        elif 'lines_or_id' in elem:
            length=len(elem)-11 # 11 is length of string 'lines_or_id'
            elem7.append(int(elem[-length:]))
            ctr_line_or+=1      
        elif 'lines_ex_id' in elem:
            length=len(elem)-11  # 11 is length of string 'lines_ex_id'
            elem8.append(int(elem[-length:]))
            ctr_line_ex+=1

    #the following sets of four lines each do this:
            
    # 1. To create the list of '2's in order to create the pair as required for 
    # the four types of elements: loads, gens, lines_or, lines_ex
    # 2. To append the list of 2s to the list of 1s
    # 3. append the elements for bus bar 2 to the previously created list for busbar1
    # 4. create the pairs of (id_number, busbar number)
    busbar2=[2]*ctr_load
    busbar_loads=busbar_loads+busbar2
    elem1=elem1+elem5 #for loads
    loads=list(zip(elem1,busbar_loads))
    
    busbar2=[2]*ctr_gen
    busbar_gens=busbar_gens+busbar2
    elem2=elem2+elem6
    gens=list(zip(elem2,busbar_gens))
    
    busbar2=[2]*ctr_line_or
    busbar_lines_or=busbar_lines_or+busbar2
    elem3=elem3+ elem7
    
    lines_or=list(zip(elem3,busbar_lines_or))
    
    busbar2=[2]*ctr_line_ex
    busbar_lines_ex=busbar_lines_ex+busbar2
    elem4=elem4+elem8
    lines_ex=list(zip(elem4,busbar_lines_ex))

    # to create the dictionary elements of this manner:
    # {"loads_id" : [(id1,busbar_nr), (id2,busbar_nr), ...], 
    # "generators_id" : [(id1,busbar_nr), (id2,busbar_nr), ...],
    # "lines_or_id" : [(id1,busbar_nr), (id2,busbar_nr), ...],
    # "lines_ex_id" : [(id1,busbar_nr), (id2,busbar_nr), ...]}
    act_dict=dict(zip(keys[0:],[loads]))
    act_dict.update(zip(keys[1:],[gens]))
    act_dict.update(zip(keys[2:],[lines_or]))
    act_dict.update(zip(keys[3:],[lines_ex]))
    # create the dictionary as "set_bus": dictionary to feed as input to 
    # the grid2op method of "action_space"
    action=action_space({"set_bus":act_dict})
    return action, act_dict # act_dict mot used, but useful for debigging

def check_gamma(combinations):
  """This function is used to remove element combinations for 
  a single busbar which violate the 'One Line Constraint',(also called 
  gamma constraint) which states:
  Out of the elements connected to a bus-bar, at least one of them has to be a line. 
  Or: a bus-bar cannot have ONLY non-line elements (loads, generators) connected to it.
  The input to this function is a combination of elements, 
  e.g: [('loads_id1,generators_id1), ('line_ex_id1,generators_id1'), ('line_or_id2,generators_id1')]
  In this example, the first combination:('loads_id1,generators_id1) violates the gamma constraint. 
  Hence, it has to be removed. So the resulting set of combinations of elements that will be returned
  from this function are
  combs : [('line_ex_id1,generators_id1'), ('line_or_id2,generators_id1')] """
  for elem in combinations: 
    #the following loop is to remove combinations
    #that have ONLY two non-line elements (gamma)
    temp1=elem[0]
    temp2=elem[1]
    if((''.join(temp1)).find('loads_id') != -1 and (''.join(temp2)).find('generators_id') != -1):
      combinations.remove(elem)
  return combinations

def return_DN_actions_indices(all_actions):
  """To only list the actions which are the same as the default configuration
      (also called the "do-nothing actions")"""
  return (len(all_actions) - 1)
  
def get_obj_connect_to_subtation(sub_items : List[Tuple['str',np.array]], 
                                 disable_line: int = -1) -> \
                                Tuple[List[str], int]:
    '''
    Returns the objects connected to a subtation.
    Parameters
    ----------
    sub_items : List['str',np.array]
        The list of tuples of object types. Each tuple consists of a
        string representing he object type and and array with the indexes
        of the connected object of that type.
        Also contains an entry with string 'nb_elements'.
    disable_line : int, optional
        The index of the line to be disabled. The default is -1.
    Returns
    -------
    sub_elem : List[str]
        String representations of the objects
    sub_nb_elem : int
        Total number of connected objects.
    '''
    sub_elem = []
    #sub_nb_elem is the number of elements conncted to this particular substation
    for k,v in sub_items:
        #this loop is to create a list of all elements connected 
        #to a single substation
        #example: k='lines_or_id' and v=array([2, 3, 4]
            if k=='nb_elements': #there is a key in this dictionary called 
            # "nb_elements" which we don't want 
            # because we only want to list the names of the elements itself
                continue
            
            #Added by Matthijs on nov 18 2021:
            #Disable the line if it is to be disabled
            if k in ('lines_or_id','lines_ex_id'):
                if disable_line in v:
                    v =  np.delete(v, np.where(v == disable_line))
                    
            
            
            if np.size(v)>1: #number of elements with key k connected to sub_id
                for j in range(np.size(v)):
                    str1 = k + str(v[j])
                    sub_elem.append(str1)
                    # example: str1 = 'lines_or_id' + str(2)
            elif np.size(v)==1:
                sub_elem.append(str(k)+str(v[0])) 
    # By the end of the above loop, sub_elem will be complete

    sub_nb_elem= len(sub_elem)
    return sub_elem, sub_nb_elem   


def create_action_space(env,substation_ids=list(range(14)), disable_line = -1, return_actions_dict = False):
    """ This function will produce a list of all actions possible for the 
    substations identified in substation_ids.

    - substations_id is a list of the ids of the substations that are in scope.
    - sub_id is the id of a substation.
    - sub_elem is a list that contains all the elements connected to a 
    single substation (identified by sub_id).
    - all_actions is a list that will contain a list of all actions for all 
    substations in substation_ids. 
    - also returns a list of indices of the all_actions list which are 
    do-nothing actions.
    
    Update 21.12:
    disable_line: liintst, Optional
        A line to be disabled. If -1 none of the lines are disabled.

    Update 17.01.2022:
    return_actions_dict: bool, optional
        If True, additionally returns a list of dictionaries defining the actions.
     """
        
    nb_elements=list(env.sub_info)  #array of number of elements connected to each 
                                     #substation 
    global keys
    global action_space
                              
    action_space=env.action_space #defining action space
    keys=list(env.get_obj_connect_to(None,0).keys()) #keys returns the names used
    all_actions=[]
    all_actions_dict = [] # amass the dictionary representations of the actions here
    DN_actions_indices = []
    temp_index = 0
    for sub_id in substation_ids: # to loop through all substations
        #print("SUBSTATION NUMBER: %d" % sub_id)
        sub_elem, sub_nb_elem = get_obj_connect_to_subtation(env.get_obj_connect_to(None, sub_id).items(),
                                                            disable_line)

        #Due to line removal, object can now be connected by only a single line (i.e. removal of line 18).
        #This is illegal, and hence throws an exception.
        if sub_nb_elem<2:
                raise Exception('Network has illegal state: this is likely due to removing ' +
                                'a powerline connected to a subtation with only two connected objects.')

        # By the end of the above loop, sub_elem will be complete
        # Below, we now start to fill the all_actions list
        # there are two cases:
        # 1. if the number of elements connected is odd
        # 2. if it is even
      
        if(sub_nb_elem%2): #if it is an odd number
            # From the formula created in the report, this part creates 
            # the alpha/2 term
            r=int((sub_nb_elem/2)+0.5) 
            # To choose half of total space, since it is
            # an odd number, we round it by adding 0.5
            for j in range(r,sub_nb_elem+1):  
                # choosing (0,sub_nb_elem+1) will result 
                # in alpha term but choosing (r,sub_nb_elem+1) results in alpha/2
                ##print('Choosing '+ str(j)+ ' out of '+str(sub_nb_elem)) 
                #bionomial coefficiants
                if(j==sub_nb_elem-1): # removing beta/2 term
                    # if it was entire "alpha" we would have to remove both 
                    # j==1 and j==n-1. But since we have already reduced alpha
                    # to alpha/2, we only need to choose the latter
                    continue # the terms for sub_nb_elem-1 are now excluded
                combs=list(it.combinations(sub_elem, j)) 
                # this command creates the combinations for "sub_elem"
                # this can also be explained by the line shown here:
                # #print('Choosing '+ str(j)+ ' out of '+str(sub_nb_elem))
                # so, sub_nb_elem is the length of sub_elem. 
                # e.g. if sub_nb_elem=6 and j =3, in this iteration it will be:
                # "choosing 3 out of 6"
                for each in combs:
                # this is the loop to create the action for a particular 
                # combination
                        single_action,single_action_dict=create_dictionary(each,sub_elem)
                        all_actions.append(single_action)
                        all_actions_dict.append(single_action_dict)
        else: # if it is an even number
            r=int(sub_nb_elem/2)  # To choose half of total space
            for j in range(r,sub_nb_elem+1): 
                # choosing (0,sub_nb_elem+1) will result 
                # in alpha term but choosing (r,sub_nb_elem+1) results in alpha/2
                #print('Choosing '+ str(j)+ ' out of '+str(sub_nb_elem))
                if(j==sub_nb_elem-1): # removing beta/2 term
                    continue
                combs=list(it.combinations(sub_elem, j))
               
                if(j==int(sub_nb_elem/2) and sub_id!=2):
                    # in an even number, the number of possiblities will be 
                    # 1 more than the number. e.g.: sub_nb_elem=6, so number of 
                    # possibilities is 7. Hence, the median (4th) possibility 
                    # should be divided by half to get exactly half the 
                    # number of possibilities. This is done below.
                    combs=combs[0:int(len(combs)/2)]
                    for each in combs:
                        single_action,single_action_dict=create_dictionary(each,sub_elem)
                        all_actions.append(single_action)
                        all_actions_dict.append(single_action_dict)
                elif(j==int(sub_nb_elem/2) and sub_id==2): #gamma term hardcoding
                    combs=combs[0:int(len(combs)/2)]
                    # these substrings are hard coded to exclude the 
                    # combinations as per the gamma constraint
                    sub_string1='loads_id1'
                    sub_string2='generators_id1'
                    for elem in combs:
                        if sub_string1 in elem and sub_string2 in elem:
                            combs.remove(elem)
                    for each in combs:
                        single_action,single_action_dict=create_dictionary(each,sub_elem)
                        all_actions.append(single_action)
                        all_actions_dict.append(single_action_dict)
                    
                elif(j==4) and (sub_id==1): # for gamma term hardcoding
                    # these substrings are hard coded to exclude the 
                    # combinations as per the gamma constraint
                    sub_string1='generators_id0'
                    sub_string2='loads_id0'
                    for elem in combs:
                        if (sub_string1 not in elem) and (sub_string2 not in elem):
                            combs.remove(elem)
                    for each in combs:
                        single_action,single_action_dict=create_dictionary(each,sub_elem)
                        all_actions.append(single_action)
                        all_actions_dict.append(single_action_dict)
                    
                elif(j==4) and (sub_id==5): #gamma term hardcoding
                    # these substrings are hard coded to exclude the 
                    # combinations as per the gamma constraint
                    sub_string1='generators_id2'
                    sub_string2='loads_id4'
                    for elem in combs:
                        if (sub_string1 not in elem) and (sub_string2 not in elem):
                            combs.remove(elem)
                    for each in combs:
                        single_action,single_action_dict=create_dictionary(each,sub_elem)
                        all_actions.append(single_action)
                        all_actions_dict.append(single_action_dict)
                    
                else:
                    # this case is for all other cases not described above
                    combs=list(it.combinations(sub_elem, j))
                    for each in combs:
                        single_action,single_action_dict=create_dictionary(each,sub_elem)
                        all_actions.append(single_action)
                        all_actions_dict.append(single_action_dict)
        
        ##print("len of dict all_actions_dict: ", len(all_actions_dict.keys()))    
        temp_index = return_DN_actions_indices(all_actions);
        DN_actions_indices.append(temp_index)
    # #print("all_actions dict: ",all_actions_dict[-1])
    # #print("len all actions", len(all_actions))
    if return_actions_dict:
        return all_actions, DN_actions_indices, all_actions_dict
    else:
        return all_actions, DN_actions_indices

def remove_redundant_actions(all_actions, reference_substation_indices, nb_elements,
                             all_actions_dict = None, remove_all_redundant_actions = False):      
    """
    Remove all the indices but one of the do-nothing actions/reference configs
    (reference configs = all elements connected to busbar1)
    for substations that have only one configuration.
    The one reference configuration that we leave is the do-nothing action
    for all substations (we are setting substation config that only has on config)

    Args:
        all_actions(list): A list of all grid2op actions.
        reference_substation_indices (list]): A list contaiining the indices of
                                            reference configurations.
        nb_elements (list): A list containing the number of elements in each substation.
        all_actions_dict (list): A list containing dictionaries defining each action.
        remove_all_redundant_actions (bool): If True, remove all the redundant actions. Else,
            leave one redundant action.
    """

    # Mask the indices that have 3 or less connected elements as they 
    # by defintion have only one configuration
    redundant_action_indices = np.array(reference_substation_indices)[np.argwhere(np.array(nb_elements)<=3).flatten()]
    left_do_nothing_action = redundant_action_indices[[0]] # leave one redundant action as a do nothing action

    start_id_to_remove = 0 if remove_all_redundant_actions else 1 # optionally leave one redundant action as a do nothing action
    for index in sorted(redundant_action_indices[start_id_to_remove:], reverse=True): # [1:] assumes the 0th action is the do-nothing action
        del all_actions[index]
        if all_actions_dict is not None:
            del all_actions_dict[index]

    if all_actions_dict is not None:
        return all_actions, list(left_do_nothing_action), all_actions_dict
    else:
        return all_actions, list(left_do_nothing_action)


if __name__ == '__main__':
  env = grid2op.make("rte_case14_realistic") #making the environment
  num_of_subs=env.n_sub # number of substations in this grid
  #substation_ids=range(num_of_subs) #default input for create_action_space function
  nb_elements=list(env.sub_info)  #array of number of elements connected to each 
                                #substation
  action_space=env.action_space #defining action space
  keys=list(env.get_obj_connect_to(None,0).keys()) #keys returns the names used 
  #for all element types: e.g: 'gen_id','load_id'
   
  print("Keys", keys)
  print("NB elements", nb_elements)
  print(len(all_actions))
  print("-"*40)
  print(DN_actions)

