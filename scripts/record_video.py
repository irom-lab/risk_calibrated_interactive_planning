from environments.hallway_env import HumanIntent
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image



# record video
def record_video(videnv, model, video_length=200, num_videos=3, save_snapshots=True):

    if num_videos <= 0:
        return
    snapshot_dir = '../hallway_snapshots/'
    print('Recording videos...')
    for intent in range(num_videos):
        if save_snapshots:
            snapshot_video_dir = os.path.join(snapshot_dir, f"video_{intent}")
            os.makedirs(snapshot_video_dir, exist_ok=True)
        intent=np.random.choice(5)
        videnv.env.envs[0].seed_intent(HumanIntent(intent))
        obs = videnv.reset()
        total_reward = 0
        for i in range(video_length):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = videnv.step(action)
            total_reward += rewards
            if save_snapshots:
                img_filename = f"rollout_time{i}.png"
                img_path = os.path.join(snapshot_video_dir, img_filename)
                img = videnv.env.render()
                Image.fromarray(img).save(img_path)
            if dones[0]:
                continue
        print(f"Total reward: {total_reward}")
        videnv.close()
    print('...Done')