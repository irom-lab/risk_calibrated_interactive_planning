import numpy as np
from stable_baselines3 import PPO, SAC  # , MultiModalPPO
# from sb3_contrib import RecurrentPPO
from environments.hallway_env import HallwayEnv
from environments.make_vectorized_hallway_env import make_env
import os
import time
import pandas as pd

from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from environments.save_best_training_reward_callback import SaveOnBestTrainingRewardCallback

from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from environments.hallway_env import HumanIntent

import matplotlib.pyplot as plt

from model_zoo.vlm_interface import extract_probs, lm


from record_video import record_video

from os.path import expanduser

import wandb
from wandb_osh.hooks import TriggerWandbSyncHook  # <-- New!

trigger_sync = TriggerWandbSyncHook()  # <--- New!

home = expanduser("~")

models_dir = f"../models/{int(time.time())}/"
logdir = os.path.join(home, f"PredictiveRL/logs/{int(time.time())}/")
dataframe_path = os.path.join(home, f"PredictiveRL/data/{int(time.time())}.csv")

render = True
debug = False
rgb_observation = True
online = False
# 'if __name__' Necessary for multithreading
if __name__ == ("__main__"):
    episodes = 1
    num_cpu = 1  # Number of processes to use
    max_steps = 200
    learn_steps = 12800
    save_freq = 100000
    n_iters=1000
    video_length=200
    timesteps = max_steps

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i, render=render, debug=debug,
                                  time_limit=max_steps, rgb_observation=rgb_observation) for i in range(num_cpu)])
    # env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env, logdir + "log")
    videnv = HallwayEnv(render=render, debug=debug, time_limit=max_steps, rgb_observation=rgb_observation)
    videnv = DummyVecEnv([lambda: videnv])
    # videnv = VecFrameStack(videnv, n_stack=4)
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
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir,
                n_steps=max_steps, n_epochs=2, learning_rate=1e-6, gamma=0.999, policy_kwargs=policy_kwargs)
    # Load model here


    num_cal = 100
    observation_list = []
    label_list = []
    for _ in range(num_cal):
        obs = videnv.reset()
        intent = np.random.choice(5)
        videnv.env.envs[0].seed_intent(HumanIntent(intent))
        total_reward = 0
        video_length_actual = np.random.randint(low=0, high=video_length)
        # rollout to a random timepoint
        for i in range(video_length):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = videnv.step(action)
            total_reward += rewards
            if dones[0]:
                continue
        print(f"Total reward: {total_reward}")
        observation = obs
        observation_list.append(obs)
        label_list.append(intent)
    dataset = pd.DataFrame({"context": observation_list, "label": label_list})
    dataset.to_csv(dataframe_path)

    non_conformity_score = []
    for index, row in dataset.iterrows():
        context = row[0]
        label = row[1]
        # save image
        # prompt lm
        # extract probs
        non_conformity_score.append(1 - true_label_smx)

    q_level = np.ceil((num_calibration + 1) * (1 - epsilon)) / num_calibration
    qhat = np.quantile(non_conformity_score, q_level, method='higher')
    print('Quantile value qhat:', qhat)
    print('')

    # plot histogram and quantile
    plt.figure(figsize=(6, 2))
    plt.hist(non_conformity_score, bins=30, edgecolor='k', linewidth=1)
    plt.axvline(
        x=qhat, linestyle='--', color='r', label='Quantile value'
    )
    plt.title(
        'Histogram of non-comformity scores in the calibration set'
    )
    plt.xlabel('Non-comformity score')
    plt.legend();
    plt.show()
    print('')
    print('A good predictor should have low non-comformity scores, concentrated at the left side of the figure')



