from environments.hallway_env import HumanIntent

# record video
def record_video(videnv, model, video_length=200):
    print('Recording videos...')
    for intent in range(1, 3):
        obs = videnv.reset()
        videnv.env.envs[0].seed_intent(HumanIntent(1))
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