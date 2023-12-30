import numpy as np
from stable_baselines3 import PPO, SAC #, MultiModalPPO
#from sb3_contrib import RecurrentPPO
from environments.hallway_env import HallwayEnv
from environments.make_vectorized_hallway_env import make_env
import os
import time

from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from environments.save_best_training_reward_callback import SaveOnBestTrainingRewardCallback

from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from record_video import record_video

from os.path import expanduser

from environments.pybullet_hallway_env import BulletHallwayEnv

env = BulletHallwayEnv(render=True, debug=True)
env.reset()
env.render()

for _ in range(1000):
    env.step(np.array([0, 0]))

time.sleep(30)
