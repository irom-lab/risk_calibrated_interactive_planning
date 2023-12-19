import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import datetime
import os
import time
import wandb
from os.path import expanduser
from stable_baselines3 import PPO
import argparse
import platform
from utils.general_utils import str2bool

from PIL import Image

from utils.visualization_utils import plot_pred, get_img_from_fig

from utils.intent_dataset import IntentPredictionDataset, collate_fn
from utils.training_utils import get_epoch_cost, calibrate_predictor
from model_zoo.intent_transformer import IntentFormer

def run():

    parser = argparse.ArgumentParser(prog='BulletHallwayEnv')
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--network-hidden-dim', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--log-history', type=str2bool, default=False)
    parser.add_argument('--load-model', type=str2bool, default=False)
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--model-load-path', type=str, default=None)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--use-discrete-action-space', type=str2bool, default=True)
    parser.add_argument('--learn-steps', type=int, default=100000, help="learn steps per epoch")
    parser.add_argument('--eval-episodes', type=int, default=10000, help="num rollouts for traj collection")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--val-batch-size', type=int, default=64)
    parser.add_argument('--num-videos', type=int, default=0)
    parser.add_argument('--counterfactual-policy-load-path', type=str, default=None)
    parser.add_argument('--rollout-num', type=int, default=None)
    parser.add_argument('--train-set-size', type=int, default=5)
    parser.add_argument('--min-traj-len', type=int, default=10)
    parser.add_argument('--max-pred-len', type=int, default=5)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--entropy-coeff', type=float, default=0.0)


    node = platform.node()
    if node == 'mae-majumdar-lab6' or node == "jlidard":
        home = expanduser("~")  # lab desktop
    elif node == 'mae-ani-lambda':
        home = expanduser("~")  # della fast IO file system
    else:
        home = '/scratch/gpfs/jlidard/'  # della fast IO file system

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
    counterfactual_policy_load_path = args["counterfactual_policy_load_path"]
    use_counterfactual_policy = log_history
    hide_intent = False
    rollout_num = args["rollout_num"]
    batch_size = args["batch_size"]
    val_batch_size = args["val_batch_size"]
    train_set_size = args["train_set_size"]
    debug = args["debug"]
    min_traj_len = args["min_traj_len"]
    max_pred_len = args["max_pred_len"]

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

    eval_policy = PPO.load(counterfactual_policy_load_path, device="cuda")

    mse_loss = torch.nn.MSELoss(reduction='none')
    CE_loss = torch.nn.CrossEntropyLoss(reduction='none')

    home = expanduser("~")
    logdir = os.path.join(home, f"PredictiveRL/models/predictor_{rollout_num}/")
    csv_dir = f"/home/jlidard/PredictiveRL/logs/{rollout_num}/rollouts"

    os.makedirs(logdir, exist_ok=True)

    hdim = 256
    future_horizon = max_pred_len
    num_segments = 1
    coord_dim = 3
    num_hiddens = 3
    n_head = 8
    nlayer = 6

    wandb.init(
        project="conformal_rl_prediction",
        mode="online"
    )


    params = {"traj_input_dim": 121,
              "num_hiddens": num_hiddens,
              "n_head": n_head,
              "num_transformer_encoder_layers": nlayer,
              "num_transformer_decoder_layers": nlayer,
              "coord_dim": coord_dim,
              "out_coord_dim": 2,
              "num_intent_modes": 5}

    learning_rate = 1e-4
    max_epochs = 30
    output_len = future_horizon
    diff_order = 1
    hidden_size = hdim
    num_layers = nlayer
    calibration_interval = 50
    traj_len = 200

    train_ds = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=True, max_pred=future_horizon, debug=debug, min_len=min_traj_len)
    test_ds = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=False, max_pred=future_horizon, debug=debug, min_len=min_traj_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=True, collate_fn=collate_fn)

    my_model = IntentFormer(hdim, num_segments, future_horizon, params=params).cuda()
    # my_model = TransformerModel(len(input_cols), input_length, output_length=output_len)
    optimizer = optim.Adam(my_model.parameters(), lr=1e-4)

    num_thresh = 100
    lambda_interval = 1 / num_thresh
    vis_interval = 5
    num_epochs = 100000
    num_cal = test_ds.__len__()
    epochs = []
    train_losses = []
    test_losses = []
    lambda_values = np.arange(0, 1, lambda_interval)
    for epoch in range(num_epochs):
        data_dict = {}
        my_model.train()

        if epoch % calibration_interval == 0:
            risk_metrics, calibration_img = calibrate_predictor(test_loader,
                                                                    my_model,
                                                                    eval_policy,
                                                                    lambda_values,
                                                                    num_cal=num_cal,
                                                                    traj_len=traj_len)
            data_dict.update(risk_metrics)
            for k, img in calibration_img.items():
                data_dict[k] = wandb.Image(img)
        epoch_cost_train, _, train_stats = get_epoch_cost(train_loader, optimizer, my_model, mse_loss, CE_loss, train=True)

        data_dict["train_loss"] = epoch_cost_train
        for k,v in train_stats.items():
            data_dict["train_" + k] = v


        if epoch % vis_interval == 0:
            with torch.no_grad():
                epoch_cost_val, viz_img, val_stats = get_epoch_cost(test_loader, optimizer, my_model, mse_loss, CE_loss, train=False)
            data_dict["val_loss"] = epoch_cost_val
            data_dict["example_vis"] = wandb.Image(viz_img)
            for k, v in val_stats.items():
                data_dict["val_" + k] = v



        epochs.append(epoch)
        train_losses.append(epoch_cost_train)
        test_losses.append(epoch_cost_val)

        wandb.log(data_dict)

        if epoch % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - "
                  f"Train Loss: {epoch_cost_train:.4f}")

        # if epoch < 2 or test_losses[-1] < test_losses[-2]:
        #     torch.save(my_model.state_dict(), f"{logdir}_epoch{epoch}.pth")

if __name__ == "__main__":
    run()


