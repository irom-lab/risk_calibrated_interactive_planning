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

from torch.optim.lr_scheduler import LambdaLR

from PIL import Image

from utils.visualization_utils import plot_pred, get_img_from_fig

from utils.risk_utils import get_knowno_epsilon_values
from utils.intent_dataset import IntentPredictionDataset, collate_fn, collate_fn_stack_only
from utils.training_utils import get_epoch_cost, calibrate_predictor, save_model
from models.intent_transformer import IntentFormer

from utils.training_utils import run_calibration

import yaml

import matplotlib


def get_params(traj_input_dim, num_intent):

    coord_dim = 3
    num_hiddens = 3
    n_head = 8
    nlayer = 6

    params = {"traj_input_dim": traj_input_dim,
              "num_hiddens": num_hiddens,
              "n_head": n_head,
              "num_transformer_encoder_layers": nlayer,
              "num_transformer_decoder_layers": nlayer,
              "coord_dim": coord_dim,
              "out_coord_dim": 3,
              "num_intent_modes": num_intent}

    return params

def run():

    font = {
            # 'weight': 'bold',
            'size': 26}

    matplotlib.rc('font', **font)

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--network-hidden-dim', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--log-history', type=str2bool, default=False)
    parser.add_argument('--load-model-path-hallway', type=str, default=None)
    parser.add_argument('--load-model-path-habitat', type=str, default=None)
    parser.add_argument('--render', type=str2bool, default=False)
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
    parser.add_argument('--offline', type=str2bool, default=False)
    parser.add_argument('--calibration-interval', type=int, default=5)
    parser.add_argument('--validation-interval', type=int, default=5)
    parser.add_argument('--calibration-set-size', type=int, default=500)
    parser.add_argument('--calibration-test-set-size', type=int, default=50)
    parser.add_argument('--use-habitat', type=str2bool, default=False)
    parser.add_argument('--use-vlm', type=str2bool, default=False)
    parser.add_argument('--traj-len', type=int, default=200)
    parser.add_argument('--num-intent', type=int, default=5)
    parser.add_argument('--traj-input-dim', type=int, default=121)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--load-in-8bit", type=str2bool, default=True)
    parser.add_argument("--load-in-4bit", type=str2bool, default=False)
    parser.add_argument("--use-llava", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument('--resume-lr', type=float, default=None)
    parser.add_argument('--num-temp', type=int, default=10)

    parser.add_argument('--yaml-file', type=str, default='../config/calibrate_hallway.yaml')





    node = platform.node()
    if node == 'mae-majumdar-lab6' or node == "jlidard":
        home = expanduser("~")  # lab desktop
    elif node == 'mae-ani-lambda':
        home = expanduser("~")  # della fast IO file system
    else:
        home = '/scratch/gpfs/jlidard/'  # della fast IO file system

    args_namespace = parser.parse_args()
    args = vars(args_namespace)
    yaml_file = args["yaml_file"]

    with open(yaml_file) as f:
        yaml_args = yaml.safe_load(f)

    args.update(yaml_args)

    max_steps = args["max_steps"]
    render = args["render"]
    num_cpu = args["num_envs"]
    log_history = args["log_history"]
    load_model_path_habitat = args["load_model_path_habitat"]
    load_model_path_hallway = args["load_model_path_hallway"]
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
    offline = args["offline"]
    calibration_interval = args["calibration_interval"]
    validation_interval = args["validation_interval"]
    calibration_set_size = args["calibration"]["calibration_set_size"]
    calibration_test_set_size = args["calibration"]["test_set_size"]
    entropy_coeff = args["entropy_coeff"]

    use_habitat = args['env'] == 'habitat'
    use_vlm =  args['env'] == 'vlm'  # TODO: differentiate 'bimanual transfer' and 'sorting'
    use_hallway =  args['env'] == 'hallway'
    use_bimanual = args['env'] == 'bimanual'


    num_intent= args["num_intent"]
    seed = args["seed"]
    traj_input_dim = args["prediction"]["traj_input_dim"]

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

    if use_hallway:
        eval_policy = PPO.load(args['policy']['path'], device="cuda")
    else:
        eval_policy = None

    mse_loss = torch.nn.MSELoss(reduction='none')
    CE_loss = torch.nn.CrossEntropyLoss(reduction='none')

    home = expanduser("~")
    logdir = os.path.join(home, f"PredictiveRL/models/predictor_{rollout_num}/")
    csv_dir = args["prediction"]["data_path"]
    if use_habitat:
        anchors = None
        min_traj_len = args["prediction"]["min_traj_len"]
        traj_len = 600
        load_model_path = load_model_path_habitat
        max_pred_len = 100
    elif use_vlm:
        anchors = None
        traj_len = 8
        min_traj_len = args["prediction"]["min_traj_len"]
        load_model_path = None
        max_pred_len = 1
        train_set_size = 1
    elif use_bimanual:
        anchors = None
        traj_len = 1
        min_traj_len = args["prediction"]["min_traj_len"]
        load_model_path = None
        max_pred_len = 1
        train_set_size = 1
    else:
        anchors_y = torch.linspace(-5, 5, 5)
        anchors = torch.zeros(5, 2)
        anchors[:, -1] = anchors_y
        min_traj_len = args["prediction"]["min_traj_len"]
        traj_len = 200
        load_model_path = load_model_path_hallway
        max_pred_len = 100
    traj_len = args["prediction"]["traj_len"]
    load_model_path = args["prediction"]["trained_model_path"]

    os.makedirs(logdir, exist_ok=True)

    # wandb.init(
    #     project="conformal_rl_prediction",
    #     mode="online" if not offline else "offline"
    # )

    hdim = 256
    future_horizon = max_pred_len
    num_segments = 1
    params = get_params(traj_input_dim, num_intent)
    learning_rate = 1e-4 if args["resume_lr"] is None else args["resume_lr"]
    max_epochs = 30
    output_len = future_horizon
    diff_order = 1
    hidden_size = hdim
    num_temp = args["calibration"]["num_temp"]
    delta = 0.01
    miscoverage_max = 0.4
    grid_points=args["calibration"]["grid_points"]
    alpha0s, eps_knowno = get_knowno_epsilon_values(dataset_size=calibration_set_size, miscoverage_max=args["calibration"]["max_miscoverage"], num_points=grid_points)
    print(alpha0s.shape)
    epsilons = eps_knowno
    alpha0s = alpha0s
    alpha0s_simpleset = np.linspace(0, 1, args["calibration"]["grid_points"])
    alpha1s = np.linspace(0, 1.0, len(eps_knowno)) # np.arange(0.04, 1, 0.004)[:2]
    print(epsilons)
    print(alpha0s)
    print(alpha0s_simpleset)
    min_temp = np.log10(args["calibration"]["minimum_temp"])
    initial_temp = args["calibration"]["initial_temp"]
    if use_vlm:
        temperatures = np.logspace(min_temp, 0.5, num_temp)
    elif use_bimanual:
        temperatures = np.logspace(min_temp, 0.5, num_temp)
    else:
        temperatures = np.logspace(initial_temp, min_temp, num_temp)
    # temperatures = np.linspace(1, 0, num_temp)
    if len(temperatures) > 1 and not use_habitat:
        temperatures[-1] = 0
    if use_habitat:
        temperatures[0] = .1
    # temperatures = np.logspace(-10, 0, num_temp)

    print(temperatures)
    # temperatures[0] = 1
    num_thresh = 2000
    lambda_interval = 1 / num_thresh
    lambda_values = np.linspace(0, 1, num_thresh)
    debug = True
    train_max_in_set = train_set_size

    train_ds = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=True,
                                       max_pred=future_horizon, debug=debug, min_len=traj_len,
                                       use_habitat=use_habitat, use_vlm=use_vlm, use_bimanual=use_bimanual, seed=seed,
                                       max_in_set=train_max_in_set)
    test_ds = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=False,
                                      max_pred=future_horizon,  debug=debug, min_len=traj_len,
                                      use_habitat=use_habitat, use_vlm=use_vlm, use_bimanual=use_bimanual, seed=seed,
                                      max_in_set=train_max_in_set)
    cal_ds = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=False,
                                     max_pred=future_horizon, debug=debug, min_len=traj_len,
                                     max_in_set=calibration_set_size, use_habitat=use_habitat, seed=seed, use_vlm=use_vlm,
                                     use_bimanual=use_bimanual, is_calibration=True)
    cal_ds_test = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=False,
                                     max_pred=future_horizon, debug=debug, min_len=traj_len,
                                     max_in_set=calibration_test_set_size, use_habitat=use_habitat, use_vlm=use_vlm,
                                     use_bimanual=use_bimanual, calibration_set_size=calibration_set_size,
                                     seed=seed, is_calibration_test=True)

    if use_vlm or use_bimanual:
        collate_dict = collate_fn_stack_only
    else:
        collate_dict = collate_fn

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_dict)
    test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=True, collate_fn=collate_dict)
    cal_loader = DataLoader(cal_ds, batch_size=args["calibration"]["batch_size"], shuffle=True, collate_fn=collate_dict)
    cal_loader_test = DataLoader(cal_ds_test, batch_size=args["calibration"]["batch_size"], shuffle=True, collate_fn=collate_dict)
    num_cal = cal_ds.__len__()

    if use_vlm or use_bimanual:
        # Only need to do a single iteration, since model is static

        data_dict = {}
        risk_metrics, _ = run_calibration(args_namespace,
                                cal_loader,
                                None,
                                eval_policy,
                                lambda_values,
                                temperatures,
                                num_cal,
                                traj_len,
                                min_traj_len,
                                num_intent,
                                use_habitat,
                                use_vlm,
                                use_bimanual,
                                epsilons,
                                delta,
                                alpha0s,
                                alpha0s_simpleset,
                                alpha1s,
                                data_dict,
                                logdir,
                                0,
                                equal_action_mask_dist=args["calibration"]["equal_action_mask_dist"])

        knowno_calibration_thresholds = risk_metrics["knowno_calibration_thresholds"]
        calibration_thresholds = risk_metrics["calibration_thresholds"]
        calibration_temps = risk_metrics["calibration_temps"]

        _, data_dict = run_calibration(args_namespace,
                        cal_loader_test,
                        None,
                        eval_policy,
                        lambda_values,
                        temperatures,
                        num_cal,
                        traj_len,
                        min_traj_len,
                        num_intent,
                        use_habitat,
                        use_vlm,
                        use_bimanual,
                        epsilons,
                        delta,
                        alpha0s,
                        alpha0s_simpleset,
                        alpha1s,
                        data_dict,
                        logdir,
                        0,
                        calibration_thresholds=calibration_thresholds,
                        knowno_calibration_thresholds=knowno_calibration_thresholds,
                        calibration_temps=calibration_temps,
                        draw_heatmap=args["calibration"]["draw_heatmap"],
                        test_cal=True,
                        equal_action_mask_dist=args["calibration"]["equal_action_mask_dist"])

        wandb.log(data_dict)
        return


    my_model = IntentFormer(hdim, num_segments, future_horizon, params=params).cuda()
    if load_model_path is not None:
        print(f"Loading model from {load_model_path}...")
        my_model.load_state_dict(torch.load(load_model_path))
        print("Done loading.")
    # my_model = TransformerModel(len(input_cols), input_length, output_length=output_len)
    optimizer = optim.Adam(my_model.parameters(), lr=learning_rate)
    # risk_evaluator = RiskEvaluator(self.intent, self.intent_predictor, self.threshold_values,
    #               self.epsilon_values, self.threshold_values_knowno,
    #               self.predict_interval, self.time_limit)



    def lr_lambda(epoch):
        decay_fracs = [1, 0.5, 0.25, 0.125, 0.125/2]
        if use_habitat:
            epoch_drops = [50, 100, 150, 200, 250]
        else:
            epoch_drops = [0, 30, 50, 100, 200]
        lowest_drop = 0
        i_lowest = 0
        for i, e in enumerate(epoch_drops):
            if e < epoch:
                lowest_drop = e
                i_lowest = i
        lr = decay_fracs[i_lowest]
        return lr
    lambda1 = lambda epoch: epoch // 30
    scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda])



    vis_interval = validation_interval
    num_epochs = 100000
    epochs = []
    train_losses = []
    test_losses = []
    should_calibrate = True

    for epoch in range(num_epochs):
        data_dict = {}
        my_model.train()

        if epoch % calibration_interval == 0:

            if should_calibrate:
                # Calibration phase
                risk_metrics, data_dict = run_calibration(args_namespace,
                                                  cal_loader,
                                                  my_model,
                                                  eval_policy,
                                                  lambda_values,
                                                  temperatures,
                                                  num_cal,
                                                  traj_len,
                                                  min_traj_len,
                                                  num_intent,
                                                  use_habitat,
                                                  use_vlm,
                                                  use_bimanual,
                                                  epsilons,
                                                  delta,
                                                  alpha0s,
                                                  alpha0s_simpleset,
                                                  alpha1s,
                                                  data_dict,
                                                  logdir,
                                                  epoch,
                                                  equal_action_mask_dist=args["calibration"]["equal_action_mask_dist"])
                knowno_calibration_thresholds = risk_metrics["knowno_calibration_thresholds"]
                calibration_thresholds = risk_metrics["calibration_thresholds"]
                calibration_temps = risk_metrics["calibration_temps"]

                _, data_dict = run_calibration(args_namespace,
                                cal_loader_test,
                                my_model,
                                eval_policy,
                                lambda_values,
                                temperatures,
                                num_cal,
                                traj_len,
                                min_traj_len,
                                num_intent,
                                use_habitat,
                                use_vlm,
                                use_bimanual,
                                epsilons,
                                delta,
                                alpha0s,
                                alpha0s_simpleset,
                                alpha1s,
                                data_dict,
                                logdir,
                                epoch,
                                calibration_thresholds=calibration_thresholds,
                                knowno_calibration_thresholds=knowno_calibration_thresholds,
                                calibration_temps=calibration_temps,
                                test_cal=True,
                                equal_action_mask_dist=args["calibration"]["equal_action_mask_dist"])

            save_model(my_model, use_habitat, use_vlm, epoch)


        epoch_cost_train, _, train_stats = get_epoch_cost(train_loader, optimizer, scheduler, my_model,
                                                          mse_loss, CE_loss, traj_len, min_traj_len,
                                                          future_horizon, train=True, ent_coeff=entropy_coeff,
                                                          use_habitat=use_habitat)

        data_dict["train_loss"] = epoch_cost_train
        for k,v in train_stats.items():
            data_dict["train_" + k] = v


        if epoch % vis_interval == 0:
            with torch.no_grad():
                epoch_cost_val, viz_img, val_stats = get_epoch_cost(test_loader, optimizer, scheduler,
                                                                    my_model, mse_loss, CE_loss,
                                                                    traj_len, min_traj_len,
                                                                    future_horizon, train=False, use_habitat=use_habitat)
            data_dict["val_loss"] = epoch_cost_val
            data_dict["example_vis"] = wandb.Image(viz_img)
            for k, v in val_stats.items():
                data_dict["val_" + k] = v

        data_dict["learning_rate"] = scheduler.get_last_lr()[0]

        epochs.append(epoch)
        train_losses.append(epoch_cost_train)
        test_losses.append(epoch_cost_val)

        wandb.log(data_dict)
        scheduler.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - "
                  f"Train Loss: {epoch_cost_train:.4f}")

        # if epoch < 2 or test_losses[-1] < test_losses[-2]:
        #

if __name__ == "__main__":
    run()


