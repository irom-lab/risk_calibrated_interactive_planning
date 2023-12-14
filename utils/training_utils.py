import torch
from utils.visualization_utils import plot_pred
from tqdm import tqdm
def get_epoch_cost(dataloader, optimizer, my_model, mse_loss, CE_loss, train=True):
    total_cost = 0
    ce_cost = 0
    mse_cost = 0
    cnt = 0
    dataloader_tqdm = tqdm(dataloader)
    for batch_dict in dataloader_tqdm:
        cnt += 1

        if train:
            my_model.train()
            optimizer.zero_grad()
        else:
            my_model.eval()
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

        ce_cost += ce_loss.detach().mean()
        mse_cost += lowest_mse_loss.detach().mean()

        total_cost += loss.detach()

        if train:
            loss.backward()
            optimizer.step()
        # print(loss.item())
        loss += loss.item()

    total_cost /= cnt
    ce_cost /= cnt
    mse_cost /= cnt

    img = None
    if not train:
        # get an example image for the last batch
        img = plot_pred(batch_X, robot_state_gt, batch_y, batch_z, y_pred, y_weight)

    stats_dict = {"ce_cost": ce_cost,
                  "mse_cost": mse_cost}

    return loss, img, stats_dict

def calibrate_predictor(predict_step_interval=10):


    for lam in lambdas:
        prediction_set_size[lam] = []

    cnt = 0
    dataloader_tqdm = tqdm(dataloader)
    for batch_dict in dataloader_tqdm:
        cnt += 1

        model.eval()
        batch_X = batch_dict["state_history"].cuda()
        batch_y = batch_dict["human_state_gt"].cuda()
        robot_state_gt = batch_dict["robot_state_gt"]
        batch_z = batch_dict["intent_gt"][:, -1].cuda()

        traj_len = batch_y.shape[1]
        traj_windows = torch.arange(10, traj_len, predict_step_interval)

        for endpoint in traj_windows:
            input = batch_z[:, :endpoint]


        y_pred, y_weight = my_model(batch_X)
        y_weight = y_weight.softmax(dim=-1)


    image_path = f'/home/jlidard/PredictiveRL/language_img/'
    for index, row in dataset.iterrows():
        context = row[0]
        label = row[1]
        for k in range(context.shape[0]):
            save_path = f'/home/jlidard/PredictiveRL/language_img/hallway_tmp{k}.png'
            img = Image.fromarray(context[k])
            img.save(save_path)
        response = vlm(prompt=prompt, image_path=image_path) # response_str = response.json()["choices"][0]["message"]["content"]
        probs = hallway_parse_response(response)
        probs = probs/probs.sum()
        true_label_smx = probs[label]
        # extract probs
        non_conformity_score.append(1 - true_label_smx)
        for lam in lambdas:
            pred_set = probs > lam
            risk = 1 if pred_set.sum() > 1 else 0
            prediction_set_size[lam].append(sum(pred_set))

        if index % 25 == 0:
            print(f"Done {index} of {num_calibration}.")