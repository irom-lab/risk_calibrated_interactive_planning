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

from PIL import Image

from utils.visualization_utils import plot_pred, get_img_from_fig

from utils.intent_dataset import IntentPredictionDataset, collate_fn
from utils.training_utils import get_epoch_cost
from model_zoo.intent_transformer import IntentFormer

mse_loss = torch.nn.MSELoss(reduction='none')
CE_loss = torch.nn.CrossEntropyLoss(reduction='none')

home = expanduser("~")
rollout_num = '1702317472'
logdir = os.path.join(home, f"PredictiveRL/models/predictor_{rollout_num}/")
csv_dir = f"/home/jlidard/PredictiveRL/logs/{rollout_num}/rollouts"

hdim = 256
future_horizon = 10
num_segments = 1
coord_dim = 3
num_hiddens = 3
n_head = 8
nlayer = 6

wandb.init(
    project="conformal_rl_prediction",
    mode="offline"
)


params = {"traj_input_dim": coord_dim*2 + 1,
          "num_hiddens": num_hiddens,
          "n_head": n_head,
          "num_transformer_encoder_layers": nlayer,
          "num_transformer_decoder_layers": nlayer,
          "coord_dim": coord_dim,
          "out_coord_dim": 2,
          "num_intent_modes": 6}

learning_rate = 1e-4
max_epochs = 30
output_len = future_horizon
diff_order = 1
hidden_size = hdim
num_layers = nlayer

train_set_size=5
train_ds = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=True, max_pred=future_horizon)
test_ds = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=False, max_pred=future_horizon)

batch_size = 256
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

my_model = IntentFormer(hdim, num_segments, future_horizon, params=params).cuda()
# my_model = TransformerModel(len(input_cols), input_length, output_length=output_len)
optimizer = optim.Adam(my_model.parameters(), lr=1e-4)


num_epochs = 100000
epochs = []
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    data_dict = {}
    my_model.train()

    for loader in (train_loader, test_loader):
        epoch_cost_train, _ = get_epoch_cost(train_loader, optimizer, my_model, mse_loss, CE_loss, train=True)
        epoch_cost_val, viz_img = get_epoch_cost(test_loader, optimizer, my_model, mse_loss, CE_loss, train=False)

    train_loss = epoch_cost_train / len(train_loader)
    val_loss = epoch_cost_val/len(test_loader)

    data_dict["train_loss"] = train_loss
    data_dict["val_loss"] = val_loss
    data_dict["vis/example_vis"] = viz_img


    epochs.append(epoch)
    train_losses.append(train_loss)
    test_losses.append(val_loss)

    wandb.log(data_dict)

    if epoch % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}")

    if epoch < 2 or test_losses[-1] < test_losses[-2]:
        torch.save(my_model.state_dict(), logdir)


