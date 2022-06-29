import numpy as np
from grid2op.Reward import L2RPNReward
from grid2op.dtypes import dt_float

from grid2op.Reward.BaseReward import BaseReward
class ScaledL2RPNReward(L2RPNReward):
    """
    Scaled version of L2RPNReward such that the reward falls between 0 and 1.
    Additionally -0.5 is awarded for illegal actions.
    """

    def initialize(self, env):
        self.reward_min = -0.5
        self.reward_illegal = -0.5
        self.reward_max = 1.0
        self.num_lines = env.backend.n_line


    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            res = np.sum(line_cap)/self.num_lines
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        # print(f"\t env.backend.get_line_flow(): {env.backend.get_line_flow()}")
        return res


    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        x = np.minimum(relative_flow, dt_float(1.0))
        lines_capacity_usage_score = np.maximum(dt_float(1.0) - x ** 2, np.zeros(x.shape, dtype=dt_float))
        return lines_capacity_usage_score

class CloseToOverflowReward(BaseReward): 
    """
    This reward finds all lines close to overflowing.
    Returns max reward when there is no overflow, min reward if more than one line is close to overflow
    and the mean between max and min reward if one line is close to overflow

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import CloseToOverflowReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "rte_case14_realistic"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=CloseToOverflowReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with this class (computing the penalty based on the number of overflow)

    """
    def __init__(self, max_lines=5):
        BaseReward.__init__(self)
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)
        self.max_overflowed = dt_float(max_lines)

    def initialize(self, env):
        pass
        
    def __call__(self,  action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        thermal_limits = env.backend.get_thermal_limit()
        lineflow_ratio = env.current_obs.rho

        close_to_overflow = dt_float(0.0)
        for ratio, limit in zip(lineflow_ratio, thermal_limits):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= 0.95) or ratio >= 0.975:
                close_to_overflow += dt_float(1.0)

        close_to_overflow = np.clip(close_to_overflow,
                                    dt_float(0.0), self.max_overflowed)
        reward = np.interp(close_to_overflow,
                           [dt_float(0.0), self.max_overflowed],
                           [self.reward_max, self.reward_min])
        return reward

class LinesReconnectedReward(BaseReward):
    """
    This reward computes a penalty
    based on the number of powerline that could have been reconnected (cooldown at 0.) but
    are still disconnected.

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import LinesReconnectedReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "rte_case14_realistic"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=LinesReconnectedReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the LinesReconnectedReward class

    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(-10.0)
        self.reward_max = dt_float(0.0)
        self.penalty_max_at_n_lines = dt_float(2.0)

    def __call__(self, action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get obs from env
        obs = env.get_obs()

        # All lines ids
        lines_id = np.arange(env.n_line)
        lines_id = lines_id[obs.time_before_cooldown_line == 0]

        n_penalties = dt_float(0.0)
        for line_id in lines_id:
            # Line could be reconnected but isn't
            if obs.line_status[line_id] == False:
                n_penalties += dt_float(1.0)

        max_p = self.penalty_max_at_n_lines
        n_penalties = np.clip(n_penalties, dt_float(0.0), max_p)
        r = np.interp(n_penalties,
                      [dt_float(0.0), max_p],
                      [self.reward_max, self.reward_min])
        return dt_float(r)


class DistanceReward(BaseReward):
    """
    This reward computes a penalty based on the distance of the current grid to the grid at time 0 where
    everything is connected to bus 1.

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import DistanceReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "rte_case14_realistic"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=DistanceReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the DistanceReward class

    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)


    def __call__(self, action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        # Get topo from env
        obs = env.get_obs()
        topo = obs.topo_vect

        idx = 0
        diff = dt_float(0.0)
        for n_elems_on_sub in obs.sub_info:
            # Find this substation elements range in topology vect
            sub_start = idx
            sub_end = idx + n_elems_on_sub
            current_sub_topo = topo[sub_start:sub_end]

            # Count number of elements not on bus 1
            # Because at the initial state, all elements are on bus 1
            diff += dt_float(1.0) * np.count_nonzero(current_sub_topo != 1)

            # Set index to next sub station
            idx += n_elems_on_sub

        r = np.interp(diff,
                      [dt_float(0.0), len(topo) * dt_float(1.0)],
                      [self.reward_max, self.reward_min])
        return r