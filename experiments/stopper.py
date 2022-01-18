import logging
from typing import Dict, Optional
from collections import defaultdict
from ray.tune.stopper import Stopper

logger = logging.getLogger(__name__)

class MaxNotImprovedStopper(Stopper):
    """Early stop single trials when they reached a plateau.

    When the standard deviation of the `metric` result of a trial is
    below a threshold `std`, the trial plateaued and will be stopped
    early.

    Args:
        metric (str): Metric to check for convergence
        num_iters_no_improvement (int): Number of iterations with no
            improvement before stopping the trial.
        percent_improve (float): Required percentage of improvement
            for a new metric to be considered better than the previous max.
        grace_period (int): Minimum number of timesteps before a trial
            can be early stopped.
        metric_threshold (Optional[float]):
            Minimum or maximum value the result has to exceed before it can
            be stopped early.
        mode (Optional[str]): If a `metric_threshold` argument has been
            passed, this must be one of [min, max]. Specifies if we optimize
            for a large metric (max) or a small metric (min). If max, the
            `metric_threshold` has to be exceeded, if min the value has to
            be lower than `metric_threshold` in order to early stop.
    """

    def __init__(self,
                 metric: str,
                 num_iters_no_improvement: int = 100,
                 percent_improve: float = 0.001, # 0.1 percent improvement needed
                 grace_period: int = 400,
                 metric_threshold: Optional[float] = None,
                 mode: Optional[str] = None):

        self._metric = metric
        self._mode = mode
        self._num_iters_no_improvement = num_iters_no_improvement
        self._current_max = float("-inf")
        self._iters_no_improvement = 0

        self._percent_improve = percent_improve
        self._grace_period = grace_period
        self._metric_threshold = metric_threshold

        if self._metric_threshold:
            if mode not in ["min", "max"]:
                raise ValueError(
                    f"When specifying a `metric_threshold`, the `mode` "
                    f"argument has to be one of [min, max]. "
                    f"Got: {mode}")

        self._iter = defaultdict(lambda: 0)
    
    def __call__(self, trial_id: str, result: Dict):
        metric_result = result.get(self._metric)
        # self._trial_results[trial_id].append(metric_result)
        self._iter[trial_id] += 1
        
        if metric_result*(1+ self._percent_improve) > self._current_max:
            self._current_max = metric_result
            self._iters_no_improvement = 0
        else:
            self._iters_no_improvement += 1

        # If still in grace period, do not stop yet
        if self._iter[trial_id] < self._grace_period:
            return False

        # If metric threshold value not reached, do not stop yet
        if self._metric_threshold is not None:
            if self._mode == "min" and metric_result > self._metric_threshold:
                return False
            elif self._mode == "max" and \
                    metric_result < self._metric_threshold:
                return False
        # If max has no been surpased in num_iters_no_improvement, early stop.
        if self._num_iters_no_improvement < self._iters_no_improvement:
            logger.info(f"Terminating the trial early with max value of {self._metric}: {self._current_max}")
            return True 
        return False # if else, return false 

    def stop_all(self):
        return False