from environments.snake_env import SnakeEnv

env = SnakeEnv()
env.reset()
# It will check your custom environment and output additional warnings if needed
# check_env(env)

episodes = 1

# Sanity check
print('[Debug] Stepping through env (feel free to comment out)...')
for episode in range(episodes):
    done = False
    obs = env.reset()
    for _ in range(10):  # not done:
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, done, info = env.step(random_action)
        print('reward', reward)
print('...Done debug.')

# models_dir = f"models/{int(time.time())}/"
# logdir = f"logs/{int(time.time())}/"
#
# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)
#
# if not os.path.exists(logdir):
#     os.makedirs(logdir)
#
# print('Training Policy.')
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
#
# TIMESTEPS = 10000
# iters = 0
# while True:
#     iters += 1
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
#     model.save(f"{models_dir}/{TIMESTEPS * iters}")
