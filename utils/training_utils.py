import torch
from utils.visualization_utils import plot_pred
def get_epoch_cost(dataloader, optimizer, my_model, mse_loss, CE_loss, train=True):
    loss = 0
    cnt = 0
    for batch_dict in dataloader:
        cnt += 1

        if train:
            optimizer.zero_grad()
        batch_X = batch_dict["state_history"]
        batch_y = batch_dict["human_state_gt"]
        robot_state_gt = batch_dict["robot_state_gt"]
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

        if train:
            loss.backward()
            optimizer.step()
        # print(loss.item())
        loss += loss.item()

    img = None
    if not train:
        # get an example image for the last batch
        img = plot_pred(batch_X, robot_state_gt, batch_y, batch_z, y_pred, y_weight)

    return loss, img