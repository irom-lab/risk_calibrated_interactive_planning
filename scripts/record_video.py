from environments.hallway_env import HumanIntent
import numpy as np

# record video
def record_video(videnv, model, video_length=200):
    print('Recording videos...')
    for intent in range(3):
        obs = videnv.reset()
        intent=np.random.choice(5)
        videnv.env.envs[0].seed_intent(HumanIntent(intent))
        total_reward = 0
        for i in range(video_length):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = videnv.step(action)
            total_reward += rewards
            if dones[0]:
                continue
        print(f"Total reward: {total_reward}")
        videnv.close()
    print('...Done')