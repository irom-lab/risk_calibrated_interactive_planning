import numpy as np
from stable_baselines3 import PPO, MultiModalPPO
from environments.hallway_env import HallwayEnv
from environments.make_vectorized_hallway_env import make_env
import os
import time

from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from environments.save_best_training_reward_callback import SaveOnBestTrainingRewardCallback

from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from record_video import record_video

import wandb

models_dir = f"../models/{int(time.time())}/"
logdir = f"/home/jlidard/PredictiveRL/logs/{int(time.time())}/"

render = False
debug = False
# 'if __name__' Necessary for multithreading
if __name__ == ("__main__"):
    episodes = 1
    num_cpu = 32 # Number of processes to use
    max_steps = 100
    learn_steps = max_steps*num_cpu*5
    save_freq = 100000
    n_iters=1000
    video_length=100
    timesteps = max_steps

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i, render=render, debug=debug, time_limit=max_steps) for i in range(num_cpu)])
    env = VecMonitor(env, logdir + "log")
    videnv = HallwayEnv(render=False, debug=debug, time_limit=max_steps)
    videnv = DummyVecEnv([lambda: videnv])
    videnv = VecVideoRecorder(videnv, video_folder=logdir, record_video_trigger=lambda x: True, video_length=video_length)

    wandb.init(
        project="conformal_rl",
    )

    print('Training Policy.')
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir)

    callback = SaveOnBestTrainingRewardCallback(check_freq=save_freq, log_dir=logdir)

    print("Starting training...")
    best_mean_reward = -np.Inf
    for iter in range(n_iters):
        model.learn(total_timesteps=learn_steps, tb_log_name=f"PPO", callback=callback)
        ep_info_buffer = model.ep_info_buffer
        training_dict = dict(model.logger.__dict__["name_to_value"])
        ep_mean_reward = safe_mean([ep_info["r"] for ep_info in ep_info_buffer])
        ep_mean_len = safe_mean([ep_info["l"] for ep_info in ep_info_buffer])
        training_dict["train/mean_reward"] = ep_mean_reward
        if ep_mean_reward > best_mean_reward:
            best_mean_reward = ep_mean_reward
        training_dict["train/best_mean_reward"] = best_mean_reward
        training_dict["time/num_timesteps"] = callback.num_timesteps
        training_dict["time/epochs"] = iter
        training_dict["time/mean_episode_length"] = ep_mean_len

        if iter % 10 == 0:
            record_video(videnv, model, video_length=video_length)

        wandb.log(training_dict)
    print("...Done.")
