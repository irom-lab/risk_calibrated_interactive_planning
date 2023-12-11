import numpy as np
from stable_baselines3 import PPO, SAC #, MultiModalPPO
#from sb3_contrib import RecurrentPPO
from environments.pybullet_hallway_env import BulletHallwayEnv
from environments.make_vectorized_hallway_env import make_bullet_env
import os
import time
import torch

from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from environments.save_best_training_reward_callback import SaveOnBestTrainingRewardCallback

from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from record_video import record_video
import platform

from os.path import expanduser

import wandb
from wandb_osh.hooks import TriggerWandbSyncHook  # <-- New!

trigger_sync = TriggerWandbSyncHook()  # <--- New!

node = platform.node()
if node == 'mae-majumdar-lab6' or node == "jlidard":
    home = expanduser("~")   # lab desktop
    num_cpu = 1
    max_steps = 300
    render = True
    debug = True
    online = False
    load_model = True
elif node == 'mae-ani-lambda':
    home = expanduser("~")   # della fast IO file system
    num_cpu = 128
    max_steps = 200
    render = False
    debug = False
    online = True
    load_model = False
else:
    home = '/scratch/gpfs/jlidard/'  # della fast IO file system
    num_cpu = 128
    max_steps = 300
    render = False
    debug = False
    online = False
    load_model = False

log_history = False

if load_model:
    load_path = '/home/jlidard/PredictiveRL/models/1702013145/model_best_450'
else:
    load_path = None

device = "cuda" if torch.cuda.is_available() else "cpu"

models_dir = f"{home}/PredictiveRL/models/{int(time.time())}/"
logdir = os.path.join(home, f"PredictiveRL/logs/{int(time.time())}/")

os.makedirs(models_dir, exist_ok=True)

rgb_observation = False
# 'if __name__' Necessary for multithreading
if __name__ == ("__main__"):
    episodes = 1
    if load_model:
        learn_steps = 25000
    else:
        learn_steps = 25000
    save_freq = 100000
    n_iters=100000
    video_length=max_steps
    timesteps = max_steps

    # Create the vectorized environment
    env = SubprocVecEnv([make_bullet_env(i,
                                         render=render,
                                         debug=debug,
                                         time_limit=max_steps,
                                         rgb_observation=rgb_observation,
                                         history_log_path=logdir)
                         for i in range(num_cpu)])
    env = VecMonitor(env, logdir + "log")
    videnv = BulletHallwayEnv(render=render, debug=debug, time_limit=max_steps, rgb_observation=rgb_observation)
    videnv = DummyVecEnv([lambda: videnv])
    videnv = VecVideoRecorder(videnv, video_folder=logdir, record_video_trigger=lambda x: True, video_length=video_length)

    if online:
        wandb.init(
            project="conformal_rl",
        )
    else:
        wandb.init(
            project="conformal_rl",
            mode="offline"
        )

    print('Training Policy.')
    policy_kwargs = dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]))
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir,
                n_steps=max_steps, n_epochs=10, learning_rate=1e-4, gamma=0.999, policy_kwargs=policy_kwargs,
                device=device)
    if load_path is not None:
        model = PPO.load(load_path, env=env)
    # model = SAC('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir, learning_rate=1e-4, gamma=0.999)

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

        if iter % 25 == 0:
            model.save(os.path.join(models_dir, f"epoch_{iter}"))

            if ep_mean_reward >= best_mean_reward:
                model.save(os.path.join(models_dir, f"model_best_{iter}"))

        if iter % 100 == 0:
            record_video(videnv, model, video_length=video_length, num_videos=2)


        wandb.log(training_dict)
        if not online:
            trigger_sync()  # <-- New!
    print("...Done.")
