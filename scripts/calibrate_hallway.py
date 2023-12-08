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
from os.path import expanduser
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook
from model_zoo.vlm_interface import vlm, hallway_parse_response

trigger_sync = TriggerWandbSyncHook()  # <--- New!

home = expanduser("~")

model_num = 1701988934  # Best kinematic
model_num = 1700717235  # Best RGB

loaddir = os.path.join(home, f"PredictiveRL/logs/{model_num}/best_model.zip")
logdir = os.path.join(home, f"PredictiveRL/conformal_outputs/{int(time.time())}/")
dataframe_path = os.path.join(home, f"PredictiveRL/conformal_outputs/{int(time.time())}.csv")


def plot_figures(non_conformity_score):
    epsilon = 0.15
    num_calibration = 100  # fake!
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
    plt.savefig('2d_hallway_non_conformity.png')
    print('')
    print('A good predictor should have low non-comformity scores, concentrated at the left side of the figure')



render = False
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
    env = HallwayEnv(render=render, debug=debug, time_limit=max_steps, rgb_observation=rgb_observation)

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
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir,
                n_steps=max_steps, n_epochs=2, learning_rate=1e-6, gamma=0.999, policy_kwargs=policy_kwargs)
    model.load(loaddir)


    num_calibration = 5
    observation_list = []
    label_list = []
    for _ in range(num_calibration):
        obs, _ = env.reset()
        intent = np.random.choice(5)
        env.seed_intent(HumanIntent(intent))
        total_reward = 0
        episode_length_actual = np.random.randint(low=video_length/10, high=video_length/5)
        # rollout to a random timepoint
        for i in range(episode_length_actual):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, trunc, info = env.step(action)
            total_reward += rewards
            if done:
                continue
        print(f"Total reward: {total_reward}")
        img_obs = env.render(resolution_scale=2)
        observation = img_obs
        observation_list.append(observation)
        label_list.append(intent)  #TODO(justin.lidard): add dynamic intents
    dataset = pd.DataFrame({"context": observation_list, "label": label_list})
    dataset.to_csv(dataframe_path)

    # img = Image.fromarray(observation["obs"], 'RGB')
    # img.save('try.png')

    non_conformity_score = []
    image_path = '/home/jlidard/PredictiveRL/hallway_tmp.png'
    for index, row in dataset.iterrows():
        context = row[0]
        label = row[1]
        img = Image.fromarray(context, 'RGB')
        img.save(image_path)
        response = vlm(prompt=prompt, image_path=image_path) # response_str = response.json()["choices"][0]["message"]["content"]
        probs = hallway_parse_response(response)
        true_label_smx = probs[label]/100
        # extract probs
        non_conformity_score.append(1 - true_label_smx)

        if index % 25 == 0:
            print(f"Done {index} of {num_calibration}.")

    plot_figures(non_conformity_score)

def hoeffding_bentkus(risk_values):
    pass




