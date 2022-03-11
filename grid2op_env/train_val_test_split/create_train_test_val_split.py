from random import sample


# This script creates a train, val and test split of the chronics
# based on the "diffuclty" of each chronic. The difficulty is defined as the number of actions
# taken by the greedy agent in the given scenario.
# We sort the chronic into difficulty, divide into 10 buckets and then uniformly sample
# from each bucket.

# We create train, val and test splits that contain 70%, 10% and 20% chronics respectively. 


import numpy as np
import grid2op

from evaluation.result_analysis import get_analysis_objects

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield np.array(lst[i:i + n])

# Specify the path to the greedy agent
EVAL_PATH = "/Users/blazejmanczak/Desktop/School/Year2/Thesis/runPowerNetworks/evaluation/eval_results/greedy_PPO_Grid_Gym_2e2dd_00001_1_num_workers=6,seed=42_2022-02-17_20-24-19"

if __name__ == "main":

    np.random.seed(0)

    _, action_analysis, topo_vects = get_analysis_objects(eval_path = EVAL_PATH )
    action_analysis.pop_implicit_no_op(topo_vects)

    chronics_sorted_len = dict(sorted(action_analysis.actions_topo_change.items(),
                            key=lambda x: len(x[1]), reverse=True))  #.keys()

    # Account for chronic id's being shifted by 1
    sorted_chronics = [chronic_id-1 for chronic_id in list(chronics_sorted_len.keys())]

    # Fix the split
    splits = [0.7,0.1,0.2]
    train_chronics, val_chronics, test_chronics = [], [], []
    for chunk in chunks(sorted_chronics, 100):
        train_size, val_size, test_size = [int(len(chunk) * split ) for split in splits]
        indices = np.random.permutation(len(chunk))
        train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size + val_size], indices[train_size + val_size:]

        train_chronics.append(chunk[train_indices])
        val_chronics.append(chunk[val_indices])
        test_chronics.append(chunk[test_indices])
        # val_chronics, test_chronics = chunk[train_indices], chunk[val_indices], chunk[test_indices]

    train_chronics = np.array(train_chronics).flatten()
    val_chronics = np.array(val_chronics).flatten()
    test_chronics = np.array(test_chronics).flatten()

    # Test 

    for chunk in chunks(sorted_chronics,100):
        num_from_chunk_train = 0
        num_from_chunk_val = 0
        num_from_chunk_test = 0
        for elem in chunk:
            if elem in train_chronics:
                num_from_chunk_train += 1
                assert elem not in val_chronics
                assert elem not in test_chronics
            if elem in val_chronics:
                num_from_chunk_val += 1
                assert elem not in train_chronics
                assert elem not in test_chronics
            if elem in test_chronics:
                num_from_chunk_test += 1
                assert elem not in train_chronics
                assert elem not in val_chronics
        
        assert num_from_chunk_train + num_from_chunk_val + num_from_chunk_test == len(chunk)
        assert num_from_chunk_train == 70
        assert num_from_chunk_val == 10
        assert num_from_chunk_test == 20
    
    env_train = grid2op.make("rte_case14_realistic"+"_train")
    env_val = grid2op.make("rte_case14_realistic"+"_val")
    env_test = grid2op.make("rte_case14_realistic"+"_test")


    # np.save("/Users/blazejmanczak/Desktop/School/Year2/Thesis/runPowerNetworks/grid2op_env/train_val_test_split/train_chronics.npy", train_chronics)
    # np.save("/Users/blazejmanczak/Desktop/School/Year2/Thesis/runPowerNetworks/grid2op_env/train_val_test_split/val_chronics.npy", val_chronics)
    # np.save("/Users/blazejmanczak/Desktop/School/Year2/Thesis/runPowerNetworks/grid2op_env/train_val_test_split/test_chronics.npy", test_chronics)
    