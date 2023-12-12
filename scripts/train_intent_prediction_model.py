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

from utils.intent_dataset import IntentPredictionDataset, collate_fn
from model_zoo.intent_transformer import IntentFormer

mse_loss = torch.nn.MSELoss(reduction='none')
CE_loss = torch.nn.CrossEntropyLoss(reduction='none')

csv_dir = "/home/jlidard/PredictiveRL/logs/1702317472/rollouts"

hdim = 256
future_horizon = 10
num_segments = 1
coord_dim = 3
num_hiddens = 3
n_head = 8
nlayer = 6


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
# test_ds = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=False)

batch_size = 256
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

my_model = IntentFormer(hdim, num_segments, future_horizon, params=params).cuda()
# my_model = TransformerModel(len(input_cols), input_length, output_length=output_len)
optimizer = optim.Adam(my_model.parameters(), lr=1e-4)


num_epochs = 100000
epochs = []
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    my_model.train()
    train_loss = 0

    cnt = 0
    for batch_dict in train_loader:
        cnt += 1
        optimizer.zero_grad()
        batch_X = batch_dict["state_history"]
        batch_y = batch_dict["state_gt"]
        batch_z = batch_dict["intent_gt"][:, -1]

        batch_X = batch_X.cuda()
        batch_y = batch_y.cuda()

        y_pred, y_weight = my_model(batch_X)
        y_weight = y_weight.softmax(dim=-1)

        loss_mse_list = []
        for mode in range(y_pred.shape[1]):
            loss_mse = mse_loss(y_pred[:, mode], batch_y)
            loss_mse_list.append(loss_mse)
        loss_mse = torch.stack(loss_mse_list, 1)
        loss_mse = loss_mse.mean(dim=-1).mean(dim=-1)
        lowest_mse_loss, lowest_mse_index = torch.min(loss_mse, dim=1)
        intent_index = batch_z

        ce_loss = CE_loss(y_weight, intent_index.long())

        loss = lowest_mse_loss + ce_loss
        loss = loss.mean()  # Finally, aggregate over batch

        loss.backward()
        optimizer.step()
        # print(loss.item())
        train_loss += loss.item()
        # break

    train_loss /= len(train_loader)
    # test_loss /= len(test_loader)

    epochs.append(epoch)
    train_losses.append(train_loss)
    # test_losses.append(test_loss)

    if epoch % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}")

    # if epoch < 2 or test_losses[-1] < test_losses[-2]:
    #     torch.save(my_model.state_dict(), weights_path)