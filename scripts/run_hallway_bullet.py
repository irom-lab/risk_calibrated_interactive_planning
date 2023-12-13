import numpy as np
from stable_baselines3 import PPO, SAC #, MultiModalPPO
#from sb3_contrib import RecurrentPPO
from environments.pybullet_hallway_env import BulletHallwayEnv
from environments.make_vectorized_hallway_env import make_bullet_env
import os
import time
import torch
import argparse

from utils.general_utils import str2bool

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

def run():
    parser = argparse.ArgumentParser(prog='BulletHallwayEnv')
    parser.add_argument('--log-history', type=str2bool, default=False)
    parser.add_argument('--load-model', type=str2bool, default=False)
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--model-load-path', type=str, default=None)
    parser.add_argument('--num-envs', type=int, default=1)
    trigger_sync = TriggerWandbSyncHook()  # <--- New!

    node = platform.node()
    if node == 'mae-majumdar-lab6' or node == "jlidard":
        home = expanduser("~")   # lab desktop
        max_steps = 100
        debug = False
        online = False

    elif node == 'mae-ani-lambda':
        home = expanduser("~")   # della fast IO file system
        max_steps = 200
        debug = False
        online = True
    else:
        home = '/scratch/gpfs/jlidard/'  # della fast IO file system
        max_steps = 300
        debug = False
        online = False

    args = vars(parser.parse_args())
    render = args["render"]
    num_cpu = args["num_envs"]
    log_history = args["log_history"]
    load_model = args["load_model"]
    load_path = args["model_load_path"]

    if load_model:
        load_path = '/home/jlidard/PredictiveRL/models/1702419335/epoch_300'
    else:
        load_path = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_dir = f"{home}/PredictiveRL/models/{int(time.time())}/"
    logdir = os.path.join(home, f"PredictiveRL/logs/{int(time.time())}/")
    history_log_path = logdir if log_history else None

    os.makedirs(models_dir, exist_ok=True)

    rgb_observation = False

    episodes = 1
    if log_history:
        learn_steps = 25000
    else:
        learn_steps = 100000
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
                                         history_log_path=history_log_path)
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
    policy_kwargs = dict(net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir,
                n_steps=max_steps, n_epochs=1, learning_rate=1e-4, gamma=0.999, policy_kwargs=policy_kwargs,
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

        if iter % 50 == 0 and not log_history:
            record_video(videnv, model, video_length=video_length, num_videos=2)


        wandb.log(training_dict)
        if not online:
            trigger_sync()  # <-- New!
    print("...Done.")


if __name__ == "__main__":
    run()
