from stable_baselines3 import PPO, MultiModalPPO
from environments.hallway_env import HallwayEnv
from environments.make_vectorized_hallway_env import make_env
import os
import time
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv

render = False


# 'if __name__' Necessary for macOS multithreading
if __name__ == "__main__":
    episodes = 1
    num_cpu = 4  # Number of processes to use

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i, render=render) for i in range(num_cpu)])

    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print('Training Policy.')
    model = MultiModalPPO('MultiModalPolicy', env, verbose=0, tensorboard_log=logdir)

    timesteps = 10000
    iters = 0
    n_iters = 10
    for iter in range(n_iters):
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=f"PPO")
        model.save(f"{models_dir}/{timesteps * iters}")
