import torch
from utils.visualization_utils import plot_pred
from tqdm import tqdm
import numpy as np
from PIL import Image
from utils.visualization_utils import draw_heatmap
import wandb
import os
# from models.vlm_interface import processed_probs, save_processed_probs
from time import sleep

from utils.risk_utils import hoeffding_bentkus
from models.vlm_interface import get_action_distribution_from_image
from utils.visualization_utils import (plot_prediction_set_size_versus_success,
                                       plot_prediction_success_versus_help_bound,
                                       plot_figures,
                                       plot_miscoverage_figure,
                                       plot_nonsingleton_figure)

from matplotlib import pyplot as plt
from scripts.temperature_demo import plot_bars
from models.bimanual_planner import BimanualPlanner

def save_model(my_model, use_habitat, use_vlm, epoch):
    if use_habitat:
        model_type = 'habitat'
    elif use_vlm:
        model_type = 'vlm'
    else:
        model_type = 'hallway'
    home = os.path.expanduser('~')
    save_dir = os.path.join(home, f'rcip/trained_models/{model_type}epoch{epoch}.pth')
    os.makedirs("/home/jlidard/rcip/trained_models/", exist_ok=True)
    if not use_vlm:
        print("Saving model to " + save_dir)
        torch.save(my_model.state_dict(), save_dir)
        print("...done.")

def run_calibration(args, cal_loader, my_model, eval_policy, lambda_values, temperatures,
                    num_cal, traj_len, min_traj_len, num_intent,
                    use_habitat, use_vlm, use_bimanual, epsilons, delta, alpha0s, alpha0s_simpleset, alpha1s, data_dict, logdir,
                    epoch, calibration_thresholds=None, knowno_calibration_thresholds=None, calibration_temps=None,
                    test_cal=False, draw_heatmap=False, equal_action_mask_dist=1):


    # Normal calibration
    risk_metrics, calibration_img = calibrate_predictor(args,
                                                        cal_loader,
                                                        my_model,
                                                        eval_policy,
                                                        lambda_values,
                                                        temperatures,
                                                        num_cal=num_cal,
                                                        traj_len=traj_len,
                                                        predict_step_interval=min_traj_len,
                                                        num_intent=num_intent,
                                                        use_habitat=use_habitat,
                                                        epsilons=epsilons,
                                                        alpha0s=alpha0s,
                                                        alpha0s_simpleset=alpha0s_simpleset,
                                                        alpha1s=alpha1s,
                                                        delta=delta,
                                                        use_vlm=use_vlm,
                                                        use_bimanual=use_bimanual,
                                                        calibration_thresholds=calibration_thresholds,
                                                        knowno_calibration_thresholds=knowno_calibration_thresholds,
                                                        calibration_temps=calibration_temps,
                                                        should_draw_heatmap=draw_heatmap,
                                                        test_cal=test_cal,
                                                        equal_action_mask_dist=equal_action_mask_dist)
    for k, img in calibration_img.items():
        data_dict[k] = wandb.Image(img)
    data_dict.update(risk_metrics)
    return risk_metrics, data_dict

def entropy(probs, dim=1):
    probs = probs + 1e-6
    logp = probs.log()
    plogp = probs * logp
    ent = -plogp.sum(dim)
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

        random_traj_len = np.random.randint(low=min_len, high=traj_len-5)
        if use_habitat:
            traj_start = max((random_traj_len // 100) * 100 - 1, 0)
        else:
            traj_start = 0
        batch_X = batch_dict["obs_full"][:, traj_start:random_traj_len]
        batch_y = batch_dict["human_full_traj"][:, random_traj_len:random_traj_len+max_pred]
        human_state_history = batch_dict["human_full_traj"][:, traj_start:random_traj_len]
        robot_state_gt = batch_dict["robot_full_traj"][:, random_traj_len:random_traj_len+max_pred]
        batch_z = batch_dict["intent_full"][:, random_traj_len]

        if use_habitat:
            batch_size = batch_X.shape[0]
            actions = batch_dict["all_actions"].cuda()
            num_intent = batch_dict["num_intent"][:, random_traj_len].int().max()
            anchors = batch_dict["obs_full"][:, random_traj_len, 7:7 + num_intent * 3]
            for z in range(num_intent):
                anchors[:, 3 * z:3 * z + 3] -= batch_dict["obs_full"][:, random_traj_len, 1:4]
            anchors = anchors.reshape(batch_size, -1, 3)
            intent_points = anchors.cuda()
        else:
            intent_points = None

        batch_X = batch_X.cuda()
        batch_y = batch_y.cuda()

        y_pred, y_weight = my_model(batch_X, human_state_history, intent_points)
        y_weight = y_weight.softmax(dim=-1)

        loss_mse_list = []
        time_range_ground_truth = batch_y.shape[1]
        y_pred= y_pred[:, :, :time_range_ground_truth]
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
        img = plot_pred(batch_X, robot_state_gt, batch_y, batch_z, y_pred, y_weight, use_habitat=use_habitat)

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

def fixed_sequence_testing(lambdas, pvals, delta, index_set, temperatures, bound_help=False, index_spacing=1):
    lambda_hat = []
    pvals_set = []
    union_bound_correction = len(temperatures) * len(index_set)
    for k in index_set:
        j = k
        lam = lambdas[j]
        loop_cond = pvals[j] <= delta / union_bound_correction
        while loop_cond:
            if lam not in lambda_hat:
                lambda_hat.append(lam)
                pvals_set.append(pvals[j])
                if j < len(lambdas)-1:
                    j = j + 1
                    loop_cond = pvals[j] <= 0.5
                else:
                    loop_cond = False
            else:
                loop_cond = False
    if len(lambda_hat) > 0:
        optimal_j = list(lambdas).index(max(lambda_hat))
    elif not bound_help:
        optimal_j = 0
        lambda_hat = [0]
    else:
        optimal_j = 0
        lambda_hat = []
    return lambda_hat, pvals_set, optimal_j

def construct_simple_set(x, eps=0.75):
    prediction_set = []
    visited = []
    running_sum = 0
    for ii in range(x.shape[-1]):
        jj = x.topk(ii+1).indices[-1]
        visited.append(jj)
        prediction_set.append(x[jj])
        running_sum += x[jj]
        sum = torch.Tensor(prediction_set).sum()
        if sum > eps:
            break
    prediction_set = torch.Tensor(prediction_set)
    return prediction_set, visited

def get_prediction_thresholds(seq_miscoverage_instance, seq_nonsingleton_instance, alpha1, alpha2, num_calibration,
                              lambdas, temperatures, delta, index_set, prefer_coverage=False, knowno_default_temp=1,
                              bound_help=False, index_spacing=1, cal_set_size=50):
    pval_miscoverage = hoeffding_bentkus(seq_miscoverage_instance, alpha_val=alpha1, n=num_calibration)
    pval_nonsingleton = hoeffding_bentkus(seq_nonsingleton_instance, alpha_val=alpha2, n=num_calibration)

    combined_pval = np.array([max(p1, p2) for (p1, p2) in zip(pval_miscoverage, pval_nonsingleton)])
    lambda_hat_set, pvals_set, optimal_lambda_index = fixed_sequence_testing(lambdas, combined_pval, delta, index_set,
                                                                         temperatures, index_spacing)
    valid_ltt = len(lambda_hat_set) > 0
    lambda_hat_set_miscoverage, _, _ = fixed_sequence_testing(lambdas, pval_miscoverage, delta, index_set, temperatures, bound_help,  index_spacing)
    lambda_hat_set_nonsingleton, _, _ = fixed_sequence_testing(lambdas, pval_nonsingleton, delta, index_set, temperatures, bound_help,  index_spacing)
    if prefer_coverage:
        optimal_lambda = lambda_hat_set[0] if len(lambda_hat_set) > 0 else 0
    else:
        optimal_lambda = lambda_hat_set[-1] if len(lambda_hat_set) > 0 else 0
    lambda_ub_miscoverage = lambda_hat_set_miscoverage[-1] if len(lambda_hat_set_miscoverage) > 0 else -1
    lambda_lb_nonsingleton = lambda_hat_set_nonsingleton[0] if len(lambda_hat_set_nonsingleton) > 0 else -1

    pvals = {"combined_pval": combined_pval,
             "pval_miscoverage": pval_miscoverage,
             "pval_nonsingleton": pval_nonsingleton}

    return lambda_hat_set, optimal_lambda, optimal_lambda_index, pvals, lambda_lb_nonsingleton, lambda_ub_miscoverage, valid_ltt

def calibrate_predictor(args, dataloader, model, policy_model, lambdas, temperatures, num_cal, traj_len=100, predict_step_interval=10,
                        num_intent=5, epsilons=0.15, alpha0s=0.15, alpha1s=0.15, alpha0s_simpleset=None, equal_action_mask_dist=1.1,
                        use_habitat=False, use_vlm=False, use_bimanual=False, test_cal=False, delta=0.05, entropy_thresh=1,
                        calibration_thresholds=None, knowno_calibration_thresholds=None, calibration_temps=None, i_knowno_temp=0,
                        report_metrics=True, should_draw_heatmap=True, knowno_default_temp=1, num_index=200):

    # TODO: add batching
    cnt = 0
    num_temperatures = len(temperatures)
    num_lambdas = len(lambdas)
    num_calibration = len(dataloader)
    num_epsilons = len(epsilons)
    num_alpha0s_simple = len(alpha0s_simpleset)
    dataloader_tqdm = tqdm(dataloader)
    # max_steps = traj_len // predict_step_interval
    if use_vlm:
        traj_len += 1
        entropy_thresh = 1
        traj_windows = torch.arange(predict_step_interval, traj_len, predict_step_interval)
    elif use_bimanual:
        planner = BimanualPlanner()
        entropy_thresh = 1
        traj_windows = torch.arange(0, traj_len, predict_step_interval)
    elif not use_habitat:
        traj_len -= predict_step_interval
        entropy_thresh = 1
        traj_windows = torch.arange(predict_step_interval, traj_len, predict_step_interval)
    else:
        entropy_thresh = 1
        traj_windows = torch.arange(predict_step_interval, traj_len+1, predict_step_interval)
    max_steps = len(traj_windows)
    non_conformity_score = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps))
    prediction_set_size = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps, num_lambdas))
    miscoverage_instance = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps, num_lambdas))
    nonsingleton_instance = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps, num_lambdas))

    simple_prediction_set_size = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps, num_alpha0s_simple))
    simple_miscoverage_instance = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps, num_alpha0s_simple))
    simple_nonsingleton_instance = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps, num_alpha0s_simple))

    nohelp_prediction_set_size = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps))
    nohelp_miscoverage_instance = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps))
    nohelp_nonsingleton_instance = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps))

    entropy_prediction_set_size = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps))
    entropy_miscoverage_instance = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps))
    entropy_nonsingleton_instance = -np.inf*torch.ones((num_temperatures, num_calibration, max_steps))
    batch_start_ind = 0
    if model is not None:
        model.eval()
    if num_index > len(lambdas):
        index_set = list(range(len(lambdas)))

    else:
        index_set = np.linspace(0, len(lambdas) // 1.2, num_index)
    index_set = [int(i) for i in index_set]
    index_spacing = len(lambdas) // len(index_set)

    score_logit_thresh = -1
    second_place_logit_thresh = 0.5
    p_thresh = 1

    if use_vlm:
        cal_str = "vlm_"
    elif use_bimanual:
        cal_str = "bimanual_"
    elif use_habitat:
        cal_str = "habitat_"
    else:
        cal_str = "hallway_"

    with torch.no_grad():
        for batch_dict in dataloader_tqdm:
            action_set_probs_list = []
            cnt += 1

            batch_z = batch_dict["intent_full"].cuda()
            dir_name = batch_dict["directory_name"]
            batch_size = batch_z.shape[0]
            batch_end_ind = batch_start_ind + batch_size

            if not use_vlm and not use_bimanual:

                batch_X = batch_dict["obs_full"].cuda()
                batch_pos = batch_dict["human_full_traj"].cuda()
                actions = batch_dict["all_actions"].cuda()
                batch_size = batch_X.shape[0]

                traj_len = batch_X.shape[1]



            for t, endpoint in enumerate(traj_windows):

                label = batch_z[:, t].long()
                if not use_vlm and not use_bimanual:
                    if use_habitat:
                        traj_start = max((endpoint // 100) * 100 - 1, 0)
                        num_intent = batch_dict["num_intent"][:, t].int().item()
                        anchors = batch_X[:, t, 7:7+num_intent*3]
                        for z in range(num_intent):
                            anchors[:, 3*z:3*z+3] -= batch_X[:, t, 1:4]
                        anchors = anchors.reshape(1, -1, 3)
                        intent_points = anchors.cuda()
                    else:
                        traj_start = 0
                        intent_points = None
                    input_t = batch_X[:, traj_start:endpoint]
                    input_human_pos = batch_pos[:, traj_start:endpoint]
                    y_pred, y_weight = model(input_t, input_human_pos, intent_points)

                    y_weight = y_weight[..., None] # add temperature dim
                    y_weight_temp = y_weight.repeat(1, 1, num_temperatures)
                    for i, temp in enumerate(temperatures):
                        y_weight_temp[..., i] = y_weight_temp[..., i] * temp
                    # y_weight_temp = y_weight.clone()
                    # y_weight_temp[:, y_weight.argmax(-1).item()] = temp * (y_weight )
                    y_weight_temp = y_weight_temp.softmax(dim=1)
                    if use_habitat:
                        action_set = actions[:, t, :num_intent*2].reshape(1, -1, 2)
                    else:
                        action_set = torch.zeros((batch_size, y_weight.shape[1], 2)).cuda()
                        for intent in range(5):
                            label_tmp = (intent)*torch.ones_like(label)[:, None]
                            obs_dict_tmp = make_obs_dict_hallway(input_t, label_tmp, intent)
                            counterfactual_action, _ = policy_model.predict(obs_dict_tmp, deterministic=True)
                            action_set[:, intent] = torch.Tensor(counterfactual_action).cuda()
                    # for each possible intent with nonzero measure, we look at the optimal action. If the optimal
                    # action matches other intents, we sum the probabilities
                    optimal_action = torch.gather(action_set, 1, label[:, None, None].repeat(1, 1, 2))
                    equal_action_mask = torch.norm(optimal_action - action_set, dim=-1)
                    equal_action_mask = equal_action_mask < equal_action_mask_dist
                    equal_action_mask[:, label.item()] = True
                    equal_action_mask = equal_action_mask[..., None].repeat(1, 1, num_temperatures)
                    action_set_probs = y_weight_temp * equal_action_mask
                    reduced_action_set_probs = torch.zeros_like(action_set_probs)
                    # equal_action_mask_label = equal_action_mask.clone()
                    # equal_action_mask_label = equal_action_mask_label * ~equal_action_mask_label
                    # equal_action_mask_label[:, label.item()] = True

                    # create a new action set that has at most the number of intents. The size is the number of intents
                    # minus the number of overlaps with the true intent-conditioned action.
                    num_duplicates = equal_action_mask.sum(1)[0, 0].item()-1 # same for all temps (assume 1 batch)
                    new_action_set = torch.zeros((batch_size, num_intent-num_duplicates, num_temperatures)).to(equal_action_mask.device)

                    # true label's action gets sum of probability mass for all duplicates
                    new_action_set[:, 0] = action_set_probs.sum(1)

                    # non-duplicate indices
                    non_duplicate_indices = equal_action_mask[0, :, 0]
                    non_duplicate_indices = [i for i, m in enumerate(non_duplicate_indices) if m == False]

                    new_action_set[:, 1:] = y_weight_temp[:, non_duplicate_indices]

                    action_set_probs = new_action_set

                    true_label_smx = new_action_set[:, 0]
                    non_conformity_score[:, batch_start_ind:batch_end_ind, t] = (1 - true_label_smx.permute(1, 0))

                    # reduced_action_set_probs[equal_action_mask_label] = action_set_probs.sum(1)
                    # reduced_action_set_probs[~equal_action_mask_label] = y_weight_temp[~equal_action_mask_label]
                    # reduced_action_set_probs = reduced_action_set_probs/reduced_action_set_probs.sum(1)
                    # action_set_probs = reduced_action_set_probs
                    # true_label_smx = action_set_probs[:, label].squeeze(1)  # torch.gather(probs, -1, label[:, None])
                    # # true_label_smx = y_weight_temp[:, label].squeeze(1)
                    # non_conformity_score[:, batch_start_ind:batch_end_ind, t] = (1 - true_label_smx.permute(1, 0))
                else:
                    dir = dir_name[0]

                    if use_bimanual:
                        instruction = batch_dict["instruction"][0]
                    else:
                        instruction = None
                    # print(f"Dir name: {dir}.")
                    # TODO: add batching for temperatures

                    if use_vlm:
                        save_path = os.path.join(dir, f"time_{t}")
                    else:
                        save_path = dir
                    # score = processed_probs(dir)


                    pred, score_llm, text = get_action_distribution_from_image(args, save_path, t, temperature_llm=knowno_default_temp, use_bimanual=use_bimanual, instruction=instruction)
                    # sleep(5)
                    if use_vlm:
                        correct = label.cpu().item() == score_llm[:-1].argmax().item()
                    else:
                        correct = False
                        for i, el in enumerate(label[0]):
                            if i == score_llm.argmax().item():
                                correct = True


                    if correct:
                        correct_str = "Correct."
                    else:
                        correct_str = "Incorrect."
                    # print(f"Processed VLM: {t} of {len(traj_windows)}. Dir name: {dir}. {correct_str}")
                    score = score_llm.clone().cuda()

                    # score = score[0]
                    # score = score[None].repeat(num_temperatures, 1, 1)  # [num_temp, B, M]

                    if any(score.isnan()):
                        score = torch.ones_like(score)
                    # score = score * temp
                    # score[score<-3] = -4.5
                    # score[0:3] -= score[-1]/3
                    # score[-1] = -np.inf
                    score_logits = score
                    top_label_logit = score_logits.max()
                    second_place_logit = score_logits.topk(2).values[-1]
                    true_label_logit = score_logits[label]
                    is_within_thresh = torch.abs(top_label_logit - true_label_logit) < score_logit_thresh
                    if use_vlm and top_label_logit != true_label_logit and is_within_thresh:
                        # if torch.abs(top_label_logit - second_place_logit) < second_place_logit_thresh:
                        #     label = score_logits.topk(2).indices[-1]
                        # else:
                        label = score_logits.argmax(-1)
                        # replace the label in post processing
                        # label = np.random.choice([score_logits.argmax().item(), score_logits.topk(2).indices[1].item()], p=[p_thresh, 1-p_thresh])
                        # label = torch.Tensor([label]).cuda().int()
                    score = score[..., None].repeat(1, 1, num_temperatures)
                    for i, temp in enumerate(temperatures):
                        score[..., i] = score[..., i] * temp
                    # if score.sum() == 0:
                    #     score = torch.ones_like(score)
                    if use_vlm:
                        score = score[:3].softmax(1)
                    else:
                        # TODO: Load in human position for planner
                        human_position = (t % 15) // 5

                        # TODO: intent -> actions
                        plans, actions = planner.plan(human_position)
                        true_intent = planner.get_true_label(label[0].cpu().numpy())
                        if true_intent > 25:
                            true_intent = 25

                        score = score.softmax(1)
                        temp_ind = 0
                        confidences = {}
                        for i_temp in range(len(temperatures)):
                            for k, plan_list in plans.items():
                                plan_confidence = torch.zeros(len(temperatures)).cuda()
                                for plan_index in plan_list:
                                    if plan_index < 26:
                                        plan_confidence = plan_confidence + score[:, plan_index]
                                confidences[k] = plan_confidence
                        action_set = torch.ones_like(score)
                        action_set = action_set[:, :5]
                        for i, v in enumerate(confidences.values()):
                            action_set[0, i] = v
                        true_plan = ''
                        for bin, intents in plans.items():
                            if true_intent in intents:
                                true_plan = bin
                        label = torch.Tensor([int(true_plan.split('_')[-1])-1]).cuda().long()
                        label = label
                        score = action_set


                    # print(score, label.item())
                    # score = torch.nan_to_num(score, 1/(score.shape[1]-1))
                    action_set_probs = score
                    true_label_smx = score[:, label].squeeze(1)

                    non_conformity_score[:, batch_start_ind:batch_end_ind, t] = (1 - true_label_smx).permute(1, 0)
                    # action_set_probs_list.append(score)



                action_set_probs_lambdas = action_set_probs[..., None].repeat(1, 1, 1, len(lambdas))
                true_label_smx_lambdas = true_label_smx[..., None].repeat(1, 1, len(lambdas))
                lambdas_tensor = torch.as_tensor(lambdas).to(action_set_probs.device)
                pred_set = action_set_probs_lambdas >= lambdas_tensor
                prediction_set_size[:, batch_start_ind:batch_end_ind, t, :] = pred_set.sum(1).permute(1, 0, 2)
                nonsingleton_instance[:, batch_start_ind:batch_end_ind, t, :] = (pred_set.sum(1) > 1).permute(1, 0, 2)
                miscoverage_instance[:, batch_start_ind:batch_end_ind, t, :] = (true_label_smx_lambdas < lambdas_tensor).permute(1, 0, 2)

                new_label = 0 if not use_vlm and not use_bimanual else label
                for i, temp in enumerate(temperatures):
                    for j, eps in enumerate(alpha0s_simpleset):
                        simple_pred_set, simple_indices = construct_simple_set(action_set_probs[0, :, i], eps)
                        simple_prediction_set_size[i, batch_start_ind:batch_end_ind, t, j] = simple_pred_set.shape[-1]
                        simple_nonsingleton_instance[i, batch_start_ind:batch_end_ind, t, j] = simple_pred_set.shape[-1] > 1
                        simple_miscoverage_instance[i, batch_start_ind:batch_end_ind, t, j] = (new_label not in simple_indices)

                nohelp_pred_set = action_set_probs.argmax(1)
                nohelp_prediction_set_size[:, batch_start_ind:batch_end_ind, t] = 1
                nohelp_nonsingleton_instance[:, batch_start_ind:batch_end_ind, t] = 0
                nohelp_miscoverage_instance[:, batch_start_ind:batch_end_ind, t] = (new_label != nohelp_pred_set).permute(1, 0)

                entropy_pred_set = action_set_probs.argmax(1)
                entropy_prediction_set_size[:, batch_start_ind:batch_end_ind, t] = 1
                entropy_nonsingleton_instance[:, batch_start_ind:batch_end_ind, t] = (entropy(action_set_probs) >= entropy_thresh).permute(1, 0)
                entropy_miscoverage_instance[:, batch_start_ind:batch_end_ind, t] = ((new_label != entropy_pred_set) * (entropy(action_set_probs) <= entropy_thresh)).permute(1, 0)


            # save_processed_probs(dir, action_set_probs_list)
            batch_start_ind = batch_end_ind

    if use_vlm:
        topk = 6
        indices = nohelp_miscoverage_instance.topk(topk, largest=False, dim=2).indices
        miscoverage_instance = torch.gather(miscoverage_instance, 2, indices[..., None].repeat(1, 1, 1, len(lambdas)))
        prediction_set_size = torch.gather(prediction_set_size,2, indices[..., None].repeat(1, 1, 1, len(lambdas)))
        nonsingleton_instance = torch.gather(nonsingleton_instance, 2, indices[..., None].repeat(1, 1, 1, len(lambdas)))
        non_conformity_score = torch.gather(non_conformity_score, 2, indices)

        simple_miscoverage_instance = torch.gather(simple_miscoverage_instance, 2,indices[..., None].repeat(1, 1, 1, len(alpha0s_simpleset)))
        simple_prediction_set_size = torch.gather(simple_prediction_set_size,2, indices[..., None].repeat(1, 1, 1, len(alpha0s_simpleset)))
        simple_nonsingleton_instance = torch.gather(simple_nonsingleton_instance, 2, indices[..., None].repeat(1, 1, 1, len(alpha0s_simpleset)))

        nohelp_miscoverage_instance = torch.gather(nohelp_miscoverage_instance, 2, indices)
        nohelp_prediction_set_size = torch.gather(nohelp_prediction_set_size,2, indices)
        nohelp_nonsingleton_instance = torch.gather(nohelp_nonsingleton_instance, 2, indices)

        entropy_miscoverage_instance = torch.gather(entropy_miscoverage_instance, 2, indices)
        entropy_prediction_set_size = torch.gather(entropy_prediction_set_size,2, indices)
        entropy_nonsingleton_instance = torch.gather(entropy_nonsingleton_instance, 2, indices)
    # Task level metrics
    seq_miscoverage_instance = miscoverage_instance.max(2).values.mean(1)
    seq_prediction_set_size = prediction_set_size.max(2).values.mean(1)
    seq_nonsingleton_instance = nonsingleton_instance.max(2).values.mean(1)
    seq_non_conformity_score = non_conformity_score.max(2).values
    seq_success_rate = 1 - miscoverage_instance.max(2).values + nonsingleton_instance.max(2).values
    seq_success_rate = torch.clip(seq_success_rate, max=1).mean(1)

    simple_seq_miscoverage_instance = simple_miscoverage_instance.max(2).values.mean(1)
    simple_seq_prediction_set_size = simple_prediction_set_size.max(2).values.mean(1)
    simple_seq_nonsingleton_instance = simple_nonsingleton_instance.max(2).values.mean(1)

    nohelp_seq_miscoverage_instance = nohelp_miscoverage_instance.max(2).values.mean(1)
    nohelp_seq_prediction_set_size = nohelp_prediction_set_size.max(2).values.mean(1)
    nohelp_seq_nonsingleton_instance = nohelp_nonsingleton_instance.max(2).values.mean(1)

    entropy_seq_miscoverage_instance = entropy_miscoverage_instance.max(2).values.mean(1)
    entropy_seq_prediction_set_size = entropy_prediction_set_size.max(2).values.mean(1)
    entropy_seq_nonsingleton_instance = entropy_nonsingleton_instance.max(2).values.mean(1)


    # Stage level metrics
    seq_miscoverage_instance_stage = miscoverage_instance.mean(2).mean(1)
    seq_nonsingleton_instance_stage = nonsingleton_instance.mean(2).mean(1)
    seq_prediction_set_size_stage = prediction_set_size.mean(2).mean(1)
    seq_non_conformity_score_stage = non_conformity_score.mean(2).values
    seq_success_rate = 1 - seq_miscoverage_instance_stage

    simple_seq_miscoverage_instance_stage = simple_miscoverage_instance.mean(2).mean(1)
    simple_seq_prediction_set_size_stage = simple_prediction_set_size.mean(2).mean(1)
    simple_seq_nonsingleton_instance_stage = simple_nonsingleton_instance.mean(2).mean(1)

    nohelp_seq_miscoverage_instance_stage = nohelp_miscoverage_instance.mean(2).mean(1)
    nohelp_seq_prediction_set_size_stage = nohelp_prediction_set_size.mean(2).mean(1)
    nohelp_seq_nonsingleton_instance_stage = nohelp_nonsingleton_instance.mean(2).mean(1)

    entropy_seq_miscoverage_instance_stage = entropy_miscoverage_instance.mean(2).mean(1)
    entropy_seq_prediction_set_size_stage = entropy_prediction_set_size.mean(2).mean(1)
    entropy_seq_nonsingleton_instance_stage = entropy_nonsingleton_instance.mean(2).mean(1)

    risk_metrics = {}
    imgs_ret = {}



    # Get the performance of the model on the calibration and test sets
    if report_metrics:
        no_help_agg_task_success_rate = np.zeros(num_epsilons)
        no_help_agg_task_success_rate_stage = np.zeros(num_epsilons)
        agg_prediction_set_size = np.zeros(num_epsilons)
        agg_task_success_rate = np.zeros(num_epsilons)
        agg_help_rate = np.zeros(num_epsilons)
        agg_task_success_rate_stage = np.zeros(num_epsilons)
        agg_help_rate_stage = np.zeros(num_epsilons)
        knowno_agg_prediction_set_size = np.zeros(num_epsilons)
        knowno_agg_task_success_rate = np.zeros(num_epsilons)
        knowno_agg_help_rate = np.zeros(num_epsilons)
        knowno_agg_task_success_rate_stage = np.zeros(num_epsilons)
        knowno_agg_help_rate_stage = np.zeros(num_epsilons)
        simple_set_agg_prediction_set_size = np.zeros(num_epsilons)
        simple_set_agg_task_success_rate = np.zeros(num_epsilons)
        simple_set_agg_help_rate = np.zeros(num_epsilons)
        simple_set_agg_task_success_rate_stage = np.zeros(num_epsilons)
        simple_set_agg_help_rate_stage = np.zeros(num_epsilons)
        entropy_set_agg_prediction_set_size = np.zeros(num_epsilons)
        entropy_set_agg_task_success_rate = np.zeros(num_epsilons)
        entropy_set_agg_help_rate = np.zeros(num_epsilons)
        entropy_set_agg_task_success_rate_stage = np.zeros(num_epsilons)
        entropy_set_agg_help_rate_stage = np.zeros(num_epsilons)
        parameter_set_sizes = np.zeros(num_epsilons)
        knowno_calibration_thresholds_new = np.zeros(num_epsilons)
        calibration_thresholds_new = np.zeros(num_epsilons)
        temp_thresholds_new = np.zeros(num_epsilons)
        best_help_rates = 2*np.ones(num_epsilons)

        # TODO: parallelize over alpha0s

        for i in range(len(alpha0s)):
            print(f"Processing: bound {i}.")

            if test_cal:
                i_knowno = knowno_calibration_thresholds[i]  # Get KnowNo threshold
                i_knowno = int(i_knowno)
                knowno_agg_prediction_set_size[i] = seq_prediction_set_size[i_knowno_temp, i_knowno].item()
                knowno_agg_task_success_rate[i] = 1-seq_miscoverage_instance[i_knowno_temp, i_knowno].item()
                knowno_agg_help_rate[i] = seq_nonsingleton_instance[i_knowno_temp, i_knowno].item()
                knowno_agg_task_success_rate_stage[i] = 1-seq_miscoverage_instance_stage[i_knowno_temp, i_knowno].item()
                knowno_agg_help_rate_stage[i] = seq_nonsingleton_instance_stage[i_knowno_temp, i_knowno].item()
                simple_set_agg_prediction_set_size[i] = simple_seq_prediction_set_size[i_knowno_temp, i].item()
                simple_set_agg_task_success_rate[i] = 1-simple_seq_miscoverage_instance[i_knowno_temp, i].item()
                simple_set_agg_help_rate[i] = simple_seq_nonsingleton_instance[i_knowno_temp, i].item()
                simple_set_agg_task_success_rate_stage[i] = 1 - simple_seq_miscoverage_instance_stage[i_knowno_temp, i].item()
                simple_set_agg_help_rate_stage[i] = simple_seq_nonsingleton_instance_stage[i_knowno_temp, i].item()
                entropy_set_agg_prediction_set_size = entropy_seq_prediction_set_size[i_knowno_temp].item()
                entropy_set_agg_task_success_rate = 1-entropy_seq_miscoverage_instance[i_knowno_temp].item()
                entropy_set_agg_help_rate = entropy_seq_nonsingleton_instance[i_knowno_temp].item()
                entropy_set_agg_task_success_rate_stage = 1 - entropy_seq_miscoverage_instance_stage[i_knowno_temp].item()
                entropy_set_agg_help_rate_stage = entropy_seq_nonsingleton_instance_stage[i_knowno_temp].item()
                no_help_agg_task_success_rate[i] = 1-nohelp_seq_miscoverage_instance[i_knowno_temp]
                no_help_agg_task_success_rate_stage[i] = 1 - nohelp_seq_miscoverage_instance_stage[i_knowno_temp]

            optimal_help_temp = []
            optimal_miscoverage_temp = []
            optimal_pset_size = []
            parameter_set_size_list = []

            for j in range(len(temperatures)):
                temp = temperatures[j]
                epsilon = epsilons[i]
                alpha0 = alpha0s[i]

                # num_calibration = 500
                q_level = min(np.ceil((num_calibration + 1) * (1 - epsilon)) / num_calibration, 1)
                qhat = 1-np.quantile(seq_non_conformity_score[j], q_level, method='higher')

                zero_crossings = np.where(np.diff(np.sign(qhat-lambdas)))[0] # find the closest lambda to knowno thresh, without going over
                i_knowno = 0 if len(zero_crossings) == 0 else zero_crossings[0]
                # i_knowno += 1


                # LTT versus knowno when coverage varies
                sol = get_prediction_thresholds(seq_miscoverage_instance[j], seq_nonsingleton_instance[j], alpha0, 1, num_calibration,
                                  lambdas, temperatures, delta, index_set, prefer_coverage=False, index_spacing=index_spacing)
                lambda_hat_set, optimal_lambda, optimal_lambda_index, pvals, lambda_lb_nonsingleton, lambda_ub_miscoverage, valid_ltt = sol
                combined_pval = pvals["combined_pval"]
                pval_miscoverage = pvals["pval_miscoverage"]
                pval_nonsingleton = pvals["pval_nonsingleton"]
                if test_cal:
                    optimal_lambda_index = calibration_thresholds[i] # Get RCIP threshold
                    i_knowno = knowno_calibration_thresholds[i] # Get KnowNo threshold
                    optimal_lambda_index = int(optimal_lambda_index)
                    i_knowno = int(i_knowno)

                print(q_level, qhat, lambdas[i_knowno], lambdas[optimal_lambda_index])

                # Sequence level: RCIP and Knowno
                optimal_nonsingleton_rate = seq_nonsingleton_instance[j, optimal_lambda_index].item()
                optimal_miscoverage_rate = seq_miscoverage_instance[j, optimal_lambda_index].item()
                optimal_prediction_set_size = seq_prediction_set_size[j, optimal_lambda_index].item()

                knowno_nonsingleton_rate = seq_nonsingleton_instance[j, i_knowno].item()
                knowno_miscoverage_rate = seq_miscoverage_instance[j, i_knowno].item()
                knowno_prediction_set_size = seq_prediction_set_size[j, i_knowno].item()

                # update thresholds
                if optimal_nonsingleton_rate < best_help_rates[i]:
                    temp_thresholds_new[i] = j
                    calibration_thresholds_new[i] = optimal_lambda_index
                    best_help_rates[i] = optimal_nonsingleton_rate


                # Stage level: RCIP and KnowNo
                optimal_nonsingleton_rate_stage = seq_nonsingleton_instance_stage[j, optimal_lambda_index].item()
                optimal_miscoverage_rate_stage = seq_miscoverage_instance_stage[j, optimal_lambda_index].item()
                optimal_prediction_set_size_stage = seq_prediction_set_size_stage[j, optimal_lambda_index].item()

                knowno_nonsingleton_rate_stage = seq_nonsingleton_instance_stage[j, i_knowno].item()
                knowno_miscoverage_rate_stage = seq_miscoverage_instance_stage[j, i_knowno].item()
                knowno_prediction_set_size_stage = seq_prediction_set_size_stage[j, i_knowno].item()

                optimal_miscoverage_temp.append((optimal_miscoverage_rate, optimal_lambda))
                optimal_help_temp.append((optimal_nonsingleton_rate, optimal_lambda))
                optimal_pset_size.append((optimal_prediction_set_size, optimal_lambda))
                parameter_set_size_list.append(len(lambda_hat_set))

                if test_cal and j == calibration_temps[i]:
                    agg_prediction_set_size[i] = optimal_prediction_set_size
                    agg_task_success_rate[i] = 1 - optimal_miscoverage_rate
                    agg_help_rate[i]  = optimal_nonsingleton_rate
                    agg_task_success_rate_stage[i] = 1 - optimal_miscoverage_rate_stage
                    agg_help_rate_stage[i]  = optimal_nonsingleton_rate_stage
                    eps_rcip_nonsingleton_rate_best_temp = optimal_nonsingleton_rate



                risk_metrics_pre = {
                    # "mean_pval_miscoverage": pval_miscoverage.mean().item(),
                    # "mean_pval_nonsingleton": pval_nonsingleton.mean().item(),
                    # "mean_combined_pval": combined_pval.mean().item(),
                    # "mean_nonsingleton_rate": seq_nonsingleton_instance.mean().item(),
                    # "mean_miscoverage_rate": seq_miscoverage_instance.mean().item(),
                    "optimal_lambda": optimal_lambda,
                    "lambda_ub_miscoverage": lambda_ub_miscoverage,
                    "lambda_lb_nonsingleton": lambda_lb_nonsingleton,
                    "optimal_pval": combined_pval[optimal_lambda_index].item(),
                    "optimal_pval_miscoverage": pval_miscoverage[optimal_lambda_index].item(),
                    "optimal_pval_nonsingleton": pval_nonsingleton[optimal_lambda_index].item(),

                    "qhat": qhat,
                    # "sequence_success_rate": seq_success_rate,
                    # "sequence_nonsingleton_instance": seq_prediction_set_size,
                    # "stage_prediction_set_size": seq_prediction_set_size_stage,
                    # "stage_non_conformity_score": seq_non_conformity_score_stage,
                    # "stage_nonsingleton_instance": seq_nonsingleton_instance_stage,
                    # "stage_success_rate": seq_success_rate,
                    "valid_ltt": 1 if valid_ltt else 0,

                    # Sequence level
                    "optimal_nonsingleton_rate": optimal_nonsingleton_rate,
                    "optimal_miscoverage_rate": optimal_miscoverage_rate,
                    "optimal_prediction_set_size": optimal_prediction_set_size,
                    "knowno_nonsingleton_rate": knowno_nonsingleton_rate,
                    "knowno_miscoverage_rate": knowno_miscoverage_rate,
                    "knowno_prediction_set_size": knowno_prediction_set_size,
                    "simple_seq_miscoverage_instance": simple_seq_miscoverage_instance,
                    "simple_seq_prediction_set_size": simple_seq_prediction_set_size,
                    "simple_seq_nonsingleton_instance": simple_seq_nonsingleton_instance,
                    "nohelp_seq_miscoverage_instance": nohelp_seq_miscoverage_instance[j],
                    "nohelp_seq_prediction_set_size": nohelp_seq_prediction_set_size[j],
                    "nohelp_seq_nonsingleton_instance": nohelp_seq_nonsingleton_instance[j],
                    "entropy_seq_miscoverage_instance": entropy_seq_miscoverage_instance[j],
                    "entropy_seq_prediction_set_size": entropy_seq_prediction_set_size[j],
                    "entropy_seq_nonsingleton_instance": entropy_seq_nonsingleton_instance[j],

                    # Instance level
                    "optimal_nonsingleton_rate_stage": optimal_nonsingleton_rate_stage,
                    "optimal_miscoverage_rate_stage": optimal_miscoverage_rate_stage,
                    "optimal_prediction_set_siz_stage": optimal_prediction_set_size_stage,
                    "knowno_nonsingleton_rate_stage": knowno_nonsingleton_rate_stage,
                    "knowno_miscoverage_rate_stage": knowno_miscoverage_rate_stage,
                    "knowno_prediction_set_size_stage": knowno_prediction_set_size_stage,
                    "simple_seq_miscoverage_instance_stage": simple_seq_miscoverage_instance_stage,
                    "simple_seq_prediction_set_size_stage": simple_seq_prediction_set_size_stage,
                    "simple_seq_nonsingleton_instance_stage": simple_seq_nonsingleton_instance_stage,
                    "nohelp_seq_miscoverage_instance_stage": nohelp_seq_miscoverage_instance_stage[j],
                    "nohelp_seq_prediction_set_size_stage": nohelp_seq_prediction_set_size_stage[j],
                    "nohelp_seq_nonsingleton_instance_stage": nohelp_seq_nonsingleton_instance_stage[j],
                    "entropy_seq_miscoverage_instance_stage": entropy_seq_miscoverage_instance_stage[j],
                    "entropy_seq_prediction_set_size_stage": entropy_seq_prediction_set_size_stage[j],
                    "entropy_seq_nonsingleton_instance_stage": entropy_seq_nonsingleton_instance_stage[j]
                }


                test_str = "test_" if test_cal else ""
                for k,v in risk_metrics_pre.items():
                    risk_metrics[f"{test_str}cal_risk_analysis_eps{alpha0}/temperature{temp}_" + k] = v

                if j == i_knowno_temp:
                    knowno_calibration_thresholds_new[i] = i_knowno
                # # get images
                if True:
                    if epsilon > 0.05 and j == i_knowno_temp and test_cal and False:

                        d = input("press enter to plot")


                        optimal_miscoverage_rcip = miscoverage_instance.max(2).values
                        optimal_miscoverage_rcip = optimal_miscoverage_rcip[int(calibration_temps[i]), ..., optimal_lambda_index]  # [N_batch]
                        prediction_set_size_rcip = prediction_set_size.max(2).values
                        prediction_set_size_rcip = prediction_set_size_rcip[int(calibration_temps[i]), ..., optimal_lambda_index]

                        num_empty_rcip = (prediction_set_size_rcip == 0).sum()
                        num_singleton_correct_rcip = ((prediction_set_size_rcip ==1 ) * (optimal_miscoverage_rcip==0)).sum()
                        num_singleton_incorrect_rcip = ((prediction_set_size_rcip ==1 ) * (optimal_miscoverage_rcip==1)).sum()
                        num_nonsingleton_correct_rcip = (
                                    (prediction_set_size_rcip > 1) * (optimal_miscoverage_rcip == 0)).sum()
                        num_nonsingleton_incorrect_rcip = (
                                    (prediction_set_size_rcip > 1) * (optimal_miscoverage_rcip == 1)).sum()
                        categories = ["E", "SC", "SI", "HC", "HI"]
                        values = [num_empty_rcip, num_singleton_correct_rcip, num_singleton_incorrect_rcip,
                                  num_nonsingleton_correct_rcip, num_nonsingleton_incorrect_rcip]
                        fig = plot_bars(categories, values, color_intent=False, default_axis=True, figsize=(16, 12),
                                        disp_int=True)
                        plt.savefig('rcip_temp.png')
                        # plt.show()


                        optimal_miscoverage_knowno = miscoverage_instance.max(2).values
                        optimal_miscoverage_knowno = optimal_miscoverage_knowno[0, ..., i_knowno]  # [N_batch]
                        prediction_set_size_knowno = prediction_set_size.max(2).values
                        prediction_set_size_knowno = prediction_set_size_knowno[0, ..., i_knowno]
                        num_empty_knowno = (prediction_set_size_knowno == 0).sum()
                        num_singleton_correct_knowno = ((prediction_set_size_knowno ==1 ) * (optimal_miscoverage_knowno==0)).sum()
                        num_singleton_incorrect_knowno = ((prediction_set_size_knowno ==1 ) * (optimal_miscoverage_knowno==1)).sum()
                        num_nonsingleton_correct_knowno = (
                                    (prediction_set_size_knowno > 1) * (optimal_miscoverage_knowno == 0)).sum()
                        num_nonsingleton_incorrect_knowno = (
                                    (prediction_set_size_knowno > 1) * (optimal_miscoverage_knowno == 1)).sum()
                        values = [num_empty_knowno, num_singleton_correct_knowno, num_singleton_incorrect_knowno,
                                  num_nonsingleton_correct_knowno, num_nonsingleton_incorrect_knowno]
                        fig = plot_bars(categories, values, color_intent=False, default_axis=True, figsize=(16, 12),
                                        disp_int=True)
                        plt.savefig('knowno_temp.png')
                        # plt.show()
                    img_miscoverage = plot_miscoverage_figure(lambdas, seq_miscoverage_instance[j], alpha=alpha0)
                    img_pred_set_size = plot_nonsingleton_figure(lambdas, seq_nonsingleton_instance[j], alpha=1)
                    img_nonconformity = plot_figures(seq_non_conformity_score[j], 1 - qhat)
                    # plt.imshow(img_nonconformity)
                    # plt.show()
                    plt.savefig('temp.png')
                    imgs = {"img_miscoverage": img_miscoverage,
                            "img_pred_set_size": img_pred_set_size,
                            "img_nonconformity": img_nonconformity}

                    for k,v in imgs.items():
                        eps_string = "%1.3f" %epsilon
                        temp_string = "%1.3f" % temp
                        pre_string = f"{test_str}cal_risk_imgs_eps{eps_string}_temperature{temp_string}/"
                        imgs_ret[pre_string + k] = v

            # TODO: aggregate over temperatures to find the best of each score per temp (nonsingleton, pred set size, coverage)
            parameter_set_sizes[i] = np.sum(parameter_set_size_list)

        risk_metrics["knowno_calibration_thresholds"] = list(knowno_calibration_thresholds_new)
        risk_metrics["calibration_thresholds"] = list(calibration_thresholds_new)
        risk_metrics["calibration_temps"] = list(temp_thresholds_new)

        # Report the performance versus ablations as coverage bound varies
        if test_cal:
            optimal_temps = [temperatures[int(i)] for i in calibration_temps]
            lb = 0.35 if use_bimanual else 0.5
            img_coverage = plot_prediction_set_size_versus_success(
                agg_prediction_set_size, agg_task_success_rate, agg_help_rate,
                knowno_agg_prediction_set_size, knowno_agg_task_success_rate, knowno_agg_help_rate,
                simple_set_agg_prediction_set_size, simple_set_agg_task_success_rate, simple_set_agg_help_rate,
                entropy_set_agg_prediction_set_size, entropy_set_agg_task_success_rate, entropy_set_agg_help_rate,
                no_help_agg_task_success_rate, optimal_temps, success_rate_lower_bound=lb
            )
            import pandas as pd
            import time
            from PIL import Image
            save_path = '/home/jlidard/risk_calibrated_interactive_planning/results'
            nano_string = "ablation_" + cal_str + str(time.time_ns())
            save_dir = os.path.join(save_path, nano_string)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_path, nano_string, 'results_ablation.pkl')
            data_dict = {
                "knowno_calibration_thresholds":  list(knowno_calibration_thresholds_new),
                "calibration_thresholds": list(calibration_thresholds_new),
                "calibration_temps": list(temp_thresholds_new),
                "agg_prediction_set_size": agg_prediction_set_size,
                "agg_task_success_rate": agg_task_success_rate,
                "agg_help_rate": agg_help_rate,
                "agg_task_success_rate_stage": agg_task_success_rate_stage,
                "agg_help_rate_stage": agg_help_rate_stage,
                "knowno_agg_prediction_set_size": knowno_agg_prediction_set_size,
                "knowno_agg_task_success_rate": knowno_agg_task_success_rate,
                "knowno_agg_help_rate": knowno_agg_help_rate,
                "knowno_agg_task_success_rate_stage": knowno_agg_task_success_rate_stage,
                "knowno_agg_help_rate_stage": knowno_agg_help_rate_stage,
                "simple_set_agg_prediction_set_size": simple_set_agg_task_success_rate,
                "simple_set_task_success_rate": simple_set_agg_task_success_rate,
                "simple_set_agg_help_rate": simple_set_agg_help_rate,
                "simple_set_task_success_rate_stage": simple_set_agg_task_success_rate_stage,
                "simple_set_agg_help_rate_stage": simple_set_agg_help_rate_stage,
                "entropy_set_agg_prediction_set_size": entropy_set_agg_prediction_set_size,
                "entropy_set_agg_task_success_rate": entropy_set_agg_task_success_rate,
                "entropy_set_agg_help_rate": entropy_set_agg_help_rate,
                "entropy_set_agg_task_success_rate_stage": entropy_set_agg_task_success_rate_stage,
                "entropy_set_agg_help_rate_stage": entropy_set_agg_help_rate_stage,
                "no_help_agg_task_success_rate": no_help_agg_task_success_rate,
                "no_help_agg_task_success_rate_stage": no_help_agg_task_success_rate_stage
            }
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(data_dict, f)
            img_coverage.save(os.path.join(save_dir,'plot.png'))
            imgs_ret[f"{test_str}cal_risk_imgs_figures/results"] = img_coverage
            img_coverage.save(os.path.join(save_dir, 'plot.png'))

            plt.imshow(img_coverage)
            plt.show()

    if should_draw_heatmap and not test_cal:


        eps_rcip_miscoverage_rate_best_temp = np.inf

        l = 0
        alpha0s_heatmap = alpha0s[::2]
        alpha1s_heatmap = alpha1s[::2]

        parameter_set_size_list = np.zeros(len(alpha0s_heatmap)*len(alpha1s_heatmap))

        for k in range(len(alpha1s_heatmap)):

            print(f"Processing: help bound {i}.")


            for i in range(len(alpha0s_heatmap)):

                parameter_set_sizes = []
                for j in range(len(temperatures)):

                    alpha0 = alpha0s_heatmap[k]
                    alpha1 = alpha1s_heatmap[i]

                    # LTT versus knowno when coverage varies
                    sol = get_prediction_thresholds(seq_miscoverage_instance[j], seq_nonsingleton_instance[j], alpha0,
                                                    alpha1, num_calibration,
                                                    lambdas, temperatures, delta, index_set, prefer_coverage=False, index_spacing=index_spacing)
                    lambda_hat_set, optimal_lambda, optimal_lambda_index, pvals, lambda_lb_nonsingleton, lambda_ub_miscoverage, valid_ltt = sol
                    combined_pval = pvals["combined_pval"]
                    pval_miscoverage = pvals["pval_miscoverage"]
                    pval_nonsingleton = pvals["pval_nonsingleton"]

                    # Sequence level: RCIP and Knowno
                    optimal_nonsingleton_rate = seq_nonsingleton_instance[j, optimal_lambda_index].item()
                    optimal_miscoverage_rate = seq_miscoverage_instance[j, optimal_lambda_index].item()
                    optimal_prediction_set_size = seq_prediction_set_size[j, optimal_lambda_index].item()

                    optimal_miscoverage_temp.append((optimal_miscoverage_rate, optimal_lambda))
                    optimal_help_temp.append((optimal_nonsingleton_rate, optimal_lambda))
                    optimal_pset_size.append((optimal_prediction_set_size, optimal_lambda))
                    parameter_set_sizes.append(len(lambda_hat_set))

                    # if optimal_miscoverage_rate < eps_rcip_miscoverage_rate_best_temp:
                    #     agg_prediction_set_size[i] = optimal_prediction_set_size
                    #     agg_task_success_rate[i] = 1 - optimal_miscoverage_rate
                    #     agg_help_rate[i] = 1 - optimal_nonsingleton_rate

                # TODO: aggregate over temperatures to find the best of each score per temp (nonsingleton, pred set size, coverage)
                parameter_set_size_list[l] = np.sum(parameter_set_sizes * len(temperatures))
                l += 1

        img_help = draw_heatmap(
            alpha0s_heatmap, alpha1s_heatmap, parameter_set_size_list
        )

        imgs_ret[f"{test_str}cal_risk_imgs_figures/results"] = img_help
        import pandas as pd
        import time
        from PIL import Image
        save_path = '/home/jlidard/PredictiveRL/results'
        nano_string = "helpbound_" + cal_str + str(time.time_ns())
        save_dir = os.path.join(save_path, nano_string)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_path, nano_string, 'results_helpbounds.pkl')
        data_dict = {
            "alpha0s_heatmap": alpha0s_heatmap,
            "alpha1s_heatmap": alpha1s_heatmap,
            "parameter_set_size_list": parameter_set_size_list
        }
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
        img_help.save(os.path.join(save_dir, 'plot.png'))
        plt.imshow(img_help)
        # plt.show()

    return risk_metrics, imgs_ret

