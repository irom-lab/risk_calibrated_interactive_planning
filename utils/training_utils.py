import torch
from utils.visualization_utils import plot_pred
from tqdm import tqdm
import numpy as np
from scripts.calibrate_hallway import plot_figures, plot_miscoverage_figure, plot_nonsingleton_figure, hoeffding_bentkus

def entropy(probs):
    probs = probs + 1e-6
    logp = probs.log()
    plogp = probs * logp
    ent = -plogp.sum(-1)
    return ent


def get_epoch_cost(dataloader, optimizer, scheduler, my_model, mse_loss, CE_loss, traj_len, min_len, max_pred,
                   train=True, mse_coeff=1, ce_coeff=100, ent_coeff=0.0, use_habitat=False):
    total_cost = 0
    ce_cost = 0
    mse_cost = 0
    entropy_cost = 0
    cnt = 0
    dataloader_tqdm = tqdm(dataloader)
    for batch_dict in dataloader_tqdm:
        cnt += 1

        if train:
            my_model.train()
            optimizer.zero_grad()
        else:
            my_model.eval()
        random_traj_len = np.random.randint(low=min_len, high=traj_len-max_pred)
        batch_X = batch_dict["obs_full"][:, :random_traj_len]
        batch_y = batch_dict["human_full_traj"][:, random_traj_len:random_traj_len+max_pred]
        human_state_history = batch_dict["human_full_traj"][:, :random_traj_len]
        robot_state_gt = batch_dict["robot_full_traj"][:, random_traj_len:random_traj_len+max_pred]
        batch_z = batch_dict["intent_full"][:, random_traj_len]

        batch_X = batch_X.cuda()
        batch_y = batch_y.cuda()

        y_pred, y_weight = my_model(batch_X, human_state_history)
        y_weight = y_weight.softmax(dim=-1)

        loss_mse_list = []
        for mode in range(y_pred.shape[1]):
            loss_mse = mse_loss(y_pred[:, mode], batch_y)
            loss_mse_list.append(loss_mse)
        loss_mse = torch.stack(loss_mse_list, 1)
        loss_mse = loss_mse.sum(dim=-1).sum(dim=-1)
        lowest_mse_loss, lowest_mse_index = torch.min(loss_mse, dim=1)
        intent_index = batch_z

        ce_loss = CE_loss(y_weight, intent_index.long())
        ent_loss = -entropy(y_weight)

        loss = mse_coeff*lowest_mse_loss + ce_coeff*ce_loss + ent_coeff*ent_loss
        loss = loss.mean()  # Finally, aggregate over batch

        ce_cost += ce_loss.detach().mean()
        mse_cost += lowest_mse_loss.detach().mean()
        entropy_cost += ent_loss.detach().mean()

        total_cost += loss.detach()

        if train:
            loss.backward()
            optimizer.step()
        # print(loss.item())

    total_cost /= cnt
    ce_cost /= cnt
    mse_cost /= cnt
    entropy_cost /= cnt

    img = None
    if not train:
        # get an example image for the last batch
        img = plot_pred(batch_X, robot_state_gt, batch_y, batch_z, y_pred, y_weight)

    stats_dict = {"ce_cost": ce_cost,
                  "mse_cost": mse_cost,
                  "entropy_cost": entropy_cost}

    return total_cost, img, stats_dict

def make_obs_dict_hallway(obs, intent, intent_index):

    batch_size, num_intent = intent.shape
    def to_onehot(z):
        onehot = torch.eye(5)[None].repeat(batch_size, 1, 1).cpu()
        intent_onehot = onehot[:, intent_index]
        return intent_onehot

    intent_onehot = to_onehot(intent)
    index_start = 1+intent_index*24
    index_end = index_start + 24
    obs_dict = {"obs": obs[..., -1, index_start:index_end].cpu(), "mode": intent_onehot}
    return obs_dict


def calibrate_predictor(dataloader, model, policy_model, lambdas, num_cal, traj_len=100, predict_step_interval=10,
                        num_intent=5, epsilon=0.15, alpha0=0.15, alpha1=0.15, equal_action_mask_dist=0.05, use_habitat=False):

    cnt = 0
    num_lambdas = len(lambdas)
    num_calibration = num_cal
    dataloader_tqdm = tqdm(dataloader)
    max_steps = traj_len // predict_step_interval
    non_conformity_score = -np.inf*torch.ones((num_calibration, max_steps))
    prediction_set_size = -np.inf*torch.ones((num_calibration, max_steps, num_lambdas))
    miscoverage_instance = -np.inf*torch.ones((num_calibration, max_steps, num_lambdas))
    batch_start_ind = 0
    model.eval()

    with torch.no_grad():
        for batch_dict in dataloader_tqdm:
            cnt += 1

            batch_X = batch_dict["obs_full"].cuda()
            batch_z = batch_dict["intent_full"].cuda()
            batch_pos = batch_dict["human_full_traj"].cuda()
            actions = batch_dict["all_actions"].cuda()
            batch_size = batch_X.shape[0]

            batch_end_ind = batch_start_ind + batch_X.shape[0]

            traj_len = batch_X.shape[1]
            traj_windows = torch.arange(predict_step_interval, traj_len, predict_step_interval)

            for t, endpoint in enumerate(traj_windows):
                input_t = batch_X[:, :endpoint]
                label = batch_z[:, endpoint-1].long()
                input_human_pos = batch_pos[:, :endpoint]
                y_pred, y_weight = model(input_t, input_human_pos)
                y_weight = y_weight.softmax(dim=-1)
                action_set = torch.zeros((batch_size, num_intent, 2)).cuda()
                for intent in range(5):
                    label_tmp = (intent)*torch.ones_like(label)[:, None]
                    if use_habitat:
                        counterfactual_action = actions[:, t]
                    else:
                        obs_dict_tmp = make_obs_dict_hallway(input_t, label_tmp, intent)
                        counterfactual_action, _ = policy_model.predict(obs_dict_tmp)
                    action_set[:, intent] = torch.Tensor(counterfactual_action).cuda()
                optimal_action = torch.gather(action_set, 1, label[:, None, None].repeat(1, 1, 2))
                equal_action_mask = torch.norm(optimal_action - action_set, dim=-1)
                equal_action_mask = equal_action_mask <= equal_action_mask_dist
                action_set_probs = y_weight * equal_action_mask
                true_label_smx = action_set_probs.sum(-1) # torch.gather(probs, -1, label[:, None])
                non_conformity_score[batch_start_ind:batch_end_ind, t] = (1 - true_label_smx).squeeze(-1)

                for i_lam, lam in enumerate(lambdas):
                    pred_set = action_set_probs > lam
                    prediction_set_size[batch_start_ind:batch_end_ind, t, i_lam] = pred_set.sum(-1) > 1
                    miscoverage_instance[batch_start_ind:batch_end_ind, t, i_lam] = (true_label_smx < lam).squeeze(-1)
            batch_start_ind = batch_end_ind

    seq_miscoverage_instance = miscoverage_instance.max(1).values.mean(0)
    seq_prediction_set_size = prediction_set_size.max(1).values.mean(0)
    seq_non_conformity_score = non_conformity_score.max(1).values

    q_level = np.ceil((num_calibration + 1) * (1 - epsilon)) / num_calibration
    qhat = np.quantile(seq_non_conformity_score, q_level, method='higher')

    pval_miscoverage = hoeffding_bentkus(seq_miscoverage_instance, alpha_val=alpha0, n=num_calibration)
    pval_nonsingleton = hoeffding_bentkus(seq_prediction_set_size, alpha_val=alpha1, n=num_calibration)
    combined_pval = np.array([max(p1, p2) for (p1, p2) in zip(pval_miscoverage, pval_nonsingleton)])
    valid_ltt = any(combined_pval < min(alpha0, alpha1))
    if valid_ltt:
        optimal_lambda_index = (combined_pval < min(alpha0, alpha1)).nonzero()[0][0].item()  # get the lowest val for optimal coverage
        highest_acceptable_pval_miscoverage_index = (pval_miscoverage < alpha0).nonzero()[-1].item()
        lowest_acceptable_pval_nonsingleton_index = (pval_nonsingleton < alpha1).nonzero()[0].item()
    else:
        optimal_lambda_index = highest_acceptable_pval_miscoverage_index = lowest_acceptable_pval_nonsingleton_index = 0
    optimal_lambda = lambdas[optimal_lambda_index]
    lambda_ub_miscoverage = lambdas[highest_acceptable_pval_miscoverage_index]
    lambda_lb_nonsingleton = lambdas[lowest_acceptable_pval_nonsingleton_index]

    img_miscoverage = plot_miscoverage_figure(lambdas, seq_miscoverage_instance, alpha=alpha0)
    img_pred_set_size = plot_nonsingleton_figure(lambdas, seq_prediction_set_size, alpha=alpha1)
    img_nonconformity = plot_figures(seq_non_conformity_score, qhat)

    imgs = {"risk_img/img_miscoverage": img_miscoverage,
            "risk_img/img_pred_set_size": img_pred_set_size,
            "risk_img/img_nonconformity":  img_nonconformity}

    risk_metrics_pre = {"mean_pval_miscoverage": pval_miscoverage.mean().item(),
                    "mean_pval_nonsingleton": pval_nonsingleton.mean().item(),
                    "mean_combined_pval": combined_pval.mean().item(),
                    "mean_nonsingleton_rate": seq_prediction_set_size.mean().item(),
                    "mean_miscoverage_rate": seq_miscoverage_instance.mean().item(),
                    "optimal_lambda": optimal_lambda,
                    "lambda_ub_miscoverage": lambda_ub_miscoverage,
                    "lambda_lb_nonsingleton": lambda_lb_nonsingleton,
                    "optimal_pval": combined_pval[optimal_lambda_index].item(),
                    "optimal_pval_miscoverage": pval_miscoverage[optimal_lambda_index].item(),
                    "optimal_pval_nonsingleton": pval_nonsingleton[optimal_lambda_index].item(),
                    "optimal_nonsingleton_rate": seq_prediction_set_size[optimal_lambda_index].item(),
                    "optimal_miscoverage_rate": seq_miscoverage_instance[optimal_lambda_index].item(),
                    "qhat": qhat,
                    "valid_ltt": 1 if valid_ltt else 0}

    risk_metrics = {}
    for k,v in risk_metrics_pre.items():
        risk_metrics["risk_analysis/" + k] = v

    return risk_metrics, imgs

