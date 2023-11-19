from stable_baselines3 import PPO, MultiModalPPO
from environments.hallway_env import HallwayEnv
from environments.make_vectorized_hallway_env import make_env
import os
import time

from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv

ender = False
# 'if __name__' Necessary for multithreading
if __name__ == ("__main__"):
    episodes = 1
    num_cpu = 1  # Number of processes to use
    max_steps = 1000
    n_iter=1

    # Create the vectorized environment
    # env = SubprocVecEnv([make_env(i, render=render) for i in range(num_cpu)])
    env = HallwayEnv(render=True, debug=True)
    target_dir = '1700259203'

    models_dir = f"../logs/{target_dir}/best_model.zip"
    logdir = f"../logs/{target_dir}/"

    print('Evaluating Policy.')
    model = PPO.load(path=models_dir, env=env)

    timesteps = max_steps*n_iter
    iters = 0
    n_iters = 1
    print("Starting evaluation...")
    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
    vec_env.close()
    print("...Done.")
