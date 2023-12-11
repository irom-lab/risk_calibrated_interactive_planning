import numpy as np
from stable_baselines3 import PPO, SAC  # , MultiModalPPO
# from sb3_contrib import RecurrentPPO
from environments.hallway_env import HallwayEnv
from environments.make_vectorized_hallway_env import make_env, make_bullet_env
import os
import time
import pandas as pd
from PIL import Image
from environments.hallway_env import HumanIntent
import matplotlib.pyplot as plt
from environments.hallway_env import HallwayEnv, prompt
from environments.pybullet_hallway_env import BulletHallwayEnv, prompt
from os.path import expanduser
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook
from model_zoo.vlm_interface import vlm, hallway_parse_response
from scipy.stats import binom

trigger_sync = TriggerWandbSyncHook()  # <--- New!

home = expanduser("~")

use_bullet = True
if not use_bullet:
    model_num = 1700717235  # Best RGB
    model_num = 1701999480  # Best kinematic
else:
    model_num = 1702013145 # 300 from 3d hallway

loaddir = os.path.join(home, f"PredictiveRL/models/{model_num}/epoch_25.zip")
logdir = os.path.join(home, f"PredictiveRL/conformal_outputs/{int(time.time())}/")
dataframe_path = os.path.join(home, f"PredictiveRL/conformal_outputs/{int(time.time())}.csv")


def plot_figures(non_conformity_score, is_bullet=False):
    epsilon = 0.15
    num_calibration = 500  # fake!
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
    plt.legend()
    if not is_bullet:
        name = 'hallway_non_conformity.png'
    else:
        name = 'bullet_hallway_non_conformity.png'
    plt.savefig(name)
    print('')
    print('A good predictor should have low non-comformity scores, concentrated at the left side of the figure')

def plot_risk_figures(prediction_set_size, is_bullet=False):

    # plot histogram and quantile
    x_list = list(prediction_set_size.keys())
    y_list = [np.mean(v) for v in prediction_set_size.values()]
    plt.figure(figsize=(6, 2))
    plt.scatter(x_list, y_list, edgecolor='k', linewidth=1)
    plt.title(
        'Prediction Set Size versus Detection Level'
    )
    plt.xlabel('Detection Level')
    plt.legend()
    if not is_bullet:
        name = 'hallway_set_size.png'
    else:
        name = 'bullet_hallway_set_size.png'
    plt.savefig(name)

# test_dict = {0: [0, 1, 2], 0.1: [1, 1, 2, 3, 5], 0.2: [1, 5, 7, 9, 11], 0.5: [0, 11], 0.10: [66]}
# plot_risk_figures(test_dict, is_bullet=True)



render = False
debug = True
rgb_observation = False
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
    if use_bullet:
        env = BulletHallwayEnv(render=render, debug=debug, time_limit=max_steps, rgb_observation=rgb_observation,show_intent=False)
    else:
        env = HallwayEnv(render=render, debug=debug, time_limit=max_steps, rgb_observation=rgb_observation ,show_intent=False)

    # if online:
    #     wandb.init(
    #         project="conformal_rl",
    #     )
    # else:
    #     wandb.init(
    #         project="conformal_rl",
    #         mode="offline"
    #     )

    print('Training Policy.')
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    # model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir,
    #             n_steps=max_steps, n_epochs=2, learning_rate=1e-6, gamma=0.999, )
    model = PPO.load(loaddir, env=env)


    num_calibration = 5
    context_list = []
    label_list = []
    for _ in range(num_calibration):
        obs, _ = env.reset()
        intent = np.random.choice(5)
        env.seed_intent(HumanIntent(intent))
        total_reward = 0
        episode_length_actual = video_length//4+5 #np.random.randint(low=video_length/10, high=video_length/5)
        observation_list = []
        # rollout to a random timepoint
        for i in range(episode_length_actual):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, trunc, info = env.step(action)
            total_reward += rewards
            if done:
                continue
            if i % 10 == 0:
                print(f"Total reward: {total_reward}")
                if use_bullet:
                    img_obs = env.render()
                else:
                    img_obs = env.render(resolution_scale=2)
                observation = img_obs
                observation_list.append(observation)
        context = np.stack(observation_list, 0)
        context_list.append(context)
        label_list.append(intent)  #TODO(justin.lidard): add dynamic intents
    dataset = pd.DataFrame({"context": context_list, "label": label_list})
    dataset.to_csv(dataframe_path)

    # img = Image.fromarray(observation["obs"], 'RGB')
    # img.save('try.png')
    env.close()
    non_conformity_score = []
    prediction_set_size = {}
    lambdas = np.arange(0, 1, 0.1)
    for lam in lambdas:
        prediction_set_size[lam] = []
    image_path = f'/home/jlidard/PredictiveRL/language_img/'
    for index, row in dataset.iterrows():
        context = row[0]
        label = row[1]
        for k in range(context.shape[0]):
            save_path = f'/home/jlidard/PredictiveRL/language_img/hallway_tmp{k}.png'
            img = Image.fromarray(context[k], 'RGB')
            img.save(save_path)
        response = vlm(prompt=prompt, image_path=image_path) # response_str = response.json()["choices"][0]["message"]["content"]
        probs = hallway_parse_response(response)
        probs = probs/probs.sum()
        true_label_smx = probs[label]
        # extract probs
        non_conformity_score.append(1 - true_label_smx)
        for lam in lambdas:
            pred_set = probs > lam
            risk = 1 if pred_set.sum() > 1 else 0
            prediction_set_size[lam].append(sum(pred_set))

        if index % 25 == 0:
            print(f"Done {index} of {num_calibration}.")

    plot_risk_figures(prediction_set_size, is_bullet=True)
    plot_figures(non_conformity_score, is_bullet=True)


def hoeffding_bentkus(risk_values, alpha_val=0.9, n=100):
    sample_risk_mean = np.mean(risk_values)

    max_alpha = np.max(sample_risk_mean, alpha_val)
    ce = cross_entropy(max_alpha, alpha_val)
    left_term = np.exp(-n * ce)

    x = np.ceil(n * sample_risk_mean)
    bin_cdf = binom.cdf(x, n, alpha_val)
    right_term = np.e * bin_cdf

    hb_p_val = np.min(left_term, right_term)
    return hb_p_val

def cross_entropy(a, b):
    return a * np.log(a/b) + (1-a) * np.log((1/a)/(1/b))




