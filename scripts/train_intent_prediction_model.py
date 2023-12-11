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

from dataset import TimeSeriesDataset
# from models import *
from losses import MSE_loss_no_batch_avg
MSE_loss = MSE_loss_no_batch_avg

from model_zoo.wind_former import WindTransformer

CE_loss = torch.nn.CrossEntropyLoss()

hdim = 256
horizon = 1000
future_horizon = 1000
num_segments = 100
coord_dim = 3
traj_input_dim = (horizon // num_segments) * coord_dim
num_hiddens = 3
n_head = 8
nlayer = 6
past_horizon = horizon


params = {"traj_input_dim": traj_input_dim,
          "num_hiddens": num_hiddens,
          "n_head": n_head,
          "num_transformer_encoder_layers": nlayer,
          "num_transformer_decoder_layers": nlayer,
          "coord_dim": coord_dim,
          "num_motion_modes": 6}

input_length = past_horizon
input_cols = ['u', 'v', 'w']
learning_rate = 1e-4
max_epochs = 30
output_len = future_horizon
diff_order = 1
hidden_size = hdim
num_layers = nlayer

train_ds = TimeSeriesDataset('sample_data.csv', input_cols, input_length, y_len=output_len, diff_order=diff_order,
                             train=True)
test_ds = TimeSeriesDataset('sample_data.csv', input_cols, input_length, y_len=output_len, diff_order=diff_order,
                            train=False)

batch_size = 256
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

my_model = WindTransformer(hdim, past_horizon, num_segments, future_horizon, params=params).cuda()
# my_model = TransformerModel(len(input_cols), input_length, output_length=output_len)
optimizer = optim.Adam(my_model.parameters(), lr=1e-4)

time = str(datetime.datetime.now())
path = "experiments/" + my_model.name + "_" + time[:-7]
path = path.replace(" ", "_")
os.system(f"mkdir {path}")
weights_path = path + "/weights.pt"
log_path = path + "/log.csv"
params_path = path + "/params.csv"
params = {"input_length": [input_length],
          "input_width": [len(input_cols)],
          "output_length": [output_len],
          "diff_order": [diff_order],
          "max_epochs": [max_epochs],
          "learning_rate": [learning_rate],
          "hidden_size": [hidden_size],
          "num_layers": [num_layers]}
params_df = pd.DataFrame(params)
params_df.to_csv(params_path)

num_epochs = 30
epochs = []
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    my_model.train()
    train_loss = 0

    print(len(train_loader))
    cnt = 0
    for batch_X, batch_y in train_loader:
        cnt += 1
        print(cnt)
        optimizer.zero_grad()

        batch_X = batch_X.cuda()
        batch_y = batch_y.cuda()

        y_pred, y_weight = my_model(batch_X)
        y_weight = y_weight[..., 0].softmax(dim=-1)
        # print("Y true:", batch_y.shape)
        # print("Y pred:", y_pred.shape)
        # print(y_pred)
        # print(y_pred.shape)
        # print(y_pred, batch_y)
        # print(y_pred.shape)
        # print(y_pred, batch_y)
        loss_mse_list = []
        for mode in range(y_pred.shape[1]):
            loss_mse = MSE_loss(y_pred[:, mode], batch_y)
            loss_mse_list.append(loss_mse)
        loss_mse = torch.stack(loss_mse_list, 1)
        loss_mse = loss_mse.mean(dim=-1).mean(dim=-1)
        lowest_mse_loss, lowest_mse_index = torch.min(loss_mse, dim=1)

        ce_loss = CE_loss(y_weight, lowest_mse_loss.long())

        loss = loss_mse + ce_loss
        loss = loss.mean()  # Finally, aggregate over batch

        loss.backward()
        optimizer.step()
        # print(loss.item())
        train_loss += loss.item()
        print(loss.item())
        # break
    # break

    test_loss = 0
    my_model.train(mode=False)
    for batch_X, batch_y in test_loader:
        y_pred, y_weight = my_model(batch_X)
        loss = MSE_loss(y_pred, batch_y)
        test_loss += loss.item()

    train_loss /= len(train_loader)
    test_loss /= len(test_loader)

    epochs.append(epoch)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}")

    if epoch < 2 or test_losses[-1] < test_losses[-2]:
        torch.save(my_model.state_dict(), weights_path)