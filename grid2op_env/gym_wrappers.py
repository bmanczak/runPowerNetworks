from gym import ObservationWrapper


class TransformObservation(ObservationWrapper):
    r"""Transform the observation via an arbitrary function.
    Example::
        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TransformObservation(env, lambda obs: obs + 0.1*np.random.randn(*obs.shape))
        >>> env.reset()
        array([-0.08319338,  0.04635121, -0.07394746,  0.20877492])
    Args:
        env (Env): environment
        f (callable): a function that transforms the observation
    """

    def __init__(self, env, f):
        super().__init__(env)
        assert callable(f)
        self.f = f
        self.grid2op_env = env.org_env

    def observation(self, observation):
        if self.grid2op_env is not None:
            return self.f(observation, self.grid2op_env)
        else:
            return self.f(observation)