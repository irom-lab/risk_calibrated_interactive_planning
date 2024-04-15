import numpy as np
from stable_baselines3 import PPO, SAC #, MultiModalPPO
#from sb3_contrib import RecurrentPPO
from environments.pybullet_hallway_env import BulletHallwayEnv
from environments.make_vectorized_hallway_env import make_bullet_env
import os
import time
import torch
import argparse

from utils.general_utils import str2bool, dict_collate

from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from environments.save_best_training_reward_callback import SaveOnBestTrainingRewardCallback

from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy

from record_video import record_video
import platform

from os.path import expanduser

import wandb
from wandb_osh.hooks import TriggerWandbSyncHook  # <-- New!

from models.intent_transformer import IntentFormer
from scripts.train_intent_prediction_model import get_params

from environments.hallway_env import HumanIntent

# record video
def deploy_conformal_policy(env, model, episode_length=200, num_videos=3):

    obs = env.reset()
    total_reward = 0
    episode_metrics = []
    for i in range(episode_length):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        metrics = {} # env.envs[0].get_metrics()
        if metrics != {}:
            episode_metrics.append(metrics)
        total_reward += rewards
        if dones[0]:
            continue

    # episode_metrics = dict_collate(episode_metrics, compute_max=True) # ensure all metrics can be taken as max (e.g. cum. reward)
    return episode_metrics

def run():
    parser = argparse.ArgumentParser(prog='BulletHallwayEnv')
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--network-hidden-dim', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--log-history', type=str2bool, default=False)
    parser.add_argument('--load-model', type=str2bool, default=False)
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--model-load-path', type=str, default=None)
    parser.add_argument('--intent-predictor-load-path', type=str, default=None)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--use-discrete-action-space', type=str2bool, default=True)
    parser.add_argument('--learn-steps', type=int, default=100000, help="learn steps per epoch")
    parser.add_argument('--eval-episodes', type=int, default=10000, help="num rollouts for traj collection")
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--num-videos', type=int, default=0)
    parser.add_argument('--hide-intent', type=str2bool, default=False)
    parser.add_argument('--counterfactual-policy-load-path', type=str, default=None)
    parser.add_argument('--record-video-only', type=str2bool, default=False)
    parser.add_argument('--wandb', type=str2bool, default=False)
    parser.add_argument('--online', type=str2bool, default=True)


    node = platform.node()
    if node == 'mae-majumdar-lab6' or node == "jlidard":
        home = expanduser("~")   # lab desktop
        debug = False
        online = False

    elif node == 'mae-ani-lambda':
        home = expanduser("~")   # della fast IO file system
        debug = False
        online = True
    else:
        home = '/scratch/gpfs/jlidard/'  # della fast IO file system
        debug = False
        online = False

    args = vars(parser.parse_args())
    max_steps = args["max_steps"]
    render = args["render"]
    num_cpu = args["num_envs"]
    log_history = args["log_history"]
    load_model = args["load_model"]
    load_path = args["model_load_path"] if load_model else None
    use_discrete_action = args["use_discrete_action_space"]
    learn_steps = args["learn_steps"]
    n_epochs = args["n_epochs"]
    hidden_dim = args["network_hidden_dim"]
    n_eval_episodes = args["eval_episodes"]
    batch_size = args["batch_size"]
    num_videos = args["num_videos"]
    record_video_only = args["record_video_only"]
    counterfactual_policy_load_path = args["counterfactual_policy_load_path"]
    use_counterfactual_policy = log_history
    intent_predictor_load_path = args["intent_predictor_load_path"]
    hide_intent = args["hide_intent"]
    online = args["online"]
    use_wandb = args["wandb"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_dir = f"{home}/PredictiveRL/models/{int(time.time())}/"
    logdir = os.path.join(home, f"PredictiveRL/logs/{int(time.time())}/")
    history_log_path = logdir if log_history else None

    os.makedirs(models_dir, exist_ok=True)

    rgb_observation = False

    episodes = 1
    save_freq = 100000
    n_iters = 100000
    video_length = max_steps
    threshold_values_knowno = np.arange(0.01, 0.25, 0.1)*3
    threshold_values = np.arange(0.01, 0.25, 0.1)*2
    epsilon_values = np.arange(0.01, 0.25, 0.1)

    # Create the vectorized environment
    if hide_intent:
        traj_input_dim = 121
        num_intent = 5
        num_segments = 1
        future_horizon = 20
        hdim = 256
        params = get_params(traj_input_dim=traj_input_dim, num_intent=num_intent)
        intent_predictor = IntentFormer(hdim, num_segments, future_horizon, params=params).cuda()
        intent_predictor.load_state_dict(torch.load(intent_predictor_load_path))
        intent_predictor.eval()
    else:
        intent_predictor = None
    env = SubprocVecEnv([make_bullet_env(i,
                                         render=render if not record_video_only else False,
                                         debug=debug,
                                         time_limit=max_steps,
                                         rgb_observation=rgb_observation,
                                         history_log_path=history_log_path,
                                         discrete_action=use_discrete_action,
                                         eval_policy=None,
                                         intent_predictor=None)
                         for i in range(num_cpu)])
    env = VecMonitor(env, logdir + "log")
    videnv = BulletHallwayEnv(render=render,
                              debug=debug,
                              time_limit=max_steps,
                              rgb_observation=rgb_observation,
                              discrete_action=use_discrete_action,
                              intent_predictor=intent_predictor,
                              threshold_values=threshold_values,
                              epsilon_values=epsilon_values,
                              threshold_values_knowno=threshold_values_knowno,
                              hide_intent=hide_intent)
    videnv = DummyVecEnv([lambda: videnv])
    if num_videos > 0:
        videnv = VecVideoRecorder(videnv, video_folder=logdir, record_video_trigger=lambda x: True, video_length=video_length)


    if use_wandb:
        if online:
            wandb.init(
                project="conformal_rl",
            )
        else:
            trigger_sync = TriggerWandbSyncHook()  # <--- New!

            wandb.init(
                project="conformal_rl",
                mode="offline"
            )


    print('Training Policy.')


    policy_kwargs = dict(net_arch=dict(pi=[hidden_dim, hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim, hidden_dim]))
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir,
                n_steps=max_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=1e-4, gamma=0.999,
                policy_kwargs=policy_kwargs, device=device)
    if load_path is not None:
        model = PPO.load(load_path, env=env)
    # model = SAC('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir, learning_rate=1e-4, gamma=0.999)

    callback = SaveOnBestTrainingRewardCallback(check_freq=save_freq, log_dir=logdir)

    print(f"Running PPO with a batch size of {batch_size}")
    print("Starting training...")
    best_mean_reward = -np.Inf
    total_metrics = []
    for iter in range(n_iters):
        # if not hide_intent:
        #     metrics = deploy_conformal_policy(videnv, model, episode_length=max_steps)
        #     total_metrics.append(metrics)
        if record_video_only:
            record_video(videnv, model, video_length=video_length, num_videos=num_videos)
        else:
            if log_history:
                evaluate_policy(model=model, env=env, n_eval_episodes=n_eval_episodes)
            else:
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

            if iter > 0 and iter % 50 == 0 and not log_history:
                record_video(videnv, model, video_length=video_length, num_videos=num_videos)

            if wandb:
                wandb.log(training_dict)
        if not online and wandb:
            trigger_sync()  # <-- New!
    total_metrics = dict_collate(total_metrics, compute_mean=True)
    wandb.log(total_metrics)
    print("...Done.")


if __name__ == "__main__":
    run()
