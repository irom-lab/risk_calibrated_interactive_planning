from environments.hallway_env import HumanIntent

# record video
def record_video(videnv, model, video_length=200):
    print('Recording videos...')
    for intent in range(1, 2):
        obs = videnv.reset()
        videnv.env.envs[0].seed_intent(HumanIntent(intent))
        for i in range(video_length):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = videnv.step(action)
            if dones[0]:
                continue
        videnv.close()
    print('...Done')