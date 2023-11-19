import gym
import numpy as np

from environments.hallway_env import HallwayEnv

def make_env(rank, seed=0, render=False,debug=False, time_limit=100):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = HallwayEnv(render=render, debug=debug, time_limit=time_limit)
        #penv.reset(seed=seed + rank)
        return env
    return _init
