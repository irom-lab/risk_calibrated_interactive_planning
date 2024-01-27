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

from utils.intent_dataset import IntentPredictionDataset, collate_fn, collate_fn_stack_only
from utils.training_utils import get_epoch_cost, calibrate_predictor
from calibrate_hallway import get_knowno_epsilon_values
from model_zoo.intent_transformer import IntentFormer

from utils.training_utils import run_calibration

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
              "out_coord_dim": 2,
              "num_intent_modes": num_intent}

    return params

def run():

    parser = argparse.ArgumentParser(prog='BulletHallwayEnv')
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
    parser.add_argument('--use-habitat', type=str2bool, default=False)
    parser.add_argument('--use-vlm', type=str2bool, default=False)
    parser.add_argument('--habitat-csv-dir', type=str, default=None)
    parser.add_argument('--traj-len', type=int, default=200)
    parser.add_argument('--num-intent', type=int, default=5)
    parser.add_argument('--traj-input-dim', type=int, default=121)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--load-in-8bit", type=str2bool, default=True)
    parser.add_argument("--load-in-4bit", type=str2bool, default=False)
    parser.add_argument("--use-llava", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument('--vlm-csv-dir', type=str, default=None)



    node = platform.node()
    if node == 'mae-majumdar-lab6' or node == "jlidard":
        home = expanduser("~")  # lab desktop
    elif node == 'mae-ani-lambda':
        home = expanduser("~")  # della fast IO file system
    else:
        home = '/scratch/gpfs/jlidard/'  # della fast IO file system

    args_namespace = parser.parse_args()
    args = vars(args_namespace)
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
    calibration_set_size = args["calibration_set_size"]
    entropy_coeff = args["entropy_coeff"]
    use_habitat = args["use_habitat"]
    use_vlm = args["use_vlm"]
    num_intent= args["num_intent"]
    seed = args["seed"]
    if use_habitat:
        traj_input_dim = 200 # args["traj_input_dim"]
    else:
        traj_input_dim = 121

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

    if not use_habitat and not use_vlm:
        eval_policy = PPO.load(counterfactual_policy_load_path, device="cuda")
    else:
        eval_policy = None

    mse_loss = torch.nn.MSELoss(reduction='none')
    CE_loss = torch.nn.CrossEntropyLoss(reduction='none')

    home = expanduser("~")
    logdir = os.path.join(home, f"PredictiveRL/models/predictor_{rollout_num}/")
    csv_dir = f"/home/jlidard/PredictiveRL/logs/{rollout_num}/rollouts"
    if use_habitat:
        csv_dir = args["habitat_csv_dir"]
        anchors = None
        min_traj_len = 100
        traj_len = 250
        load_model_path = load_model_path_habitat
    elif use_vlm:
        csv_dir = args["vlm_csv_dir"]
        anchors = None
        traj_len = 8
        load_model_path = None
    else:
        anchors_y = torch.linspace(-5, 5, 5)
        anchors = torch.zeros(5, 2)
        anchors[:, -1] = anchors_y
        min_traj_len = 20
        traj_len = 200
        load_model_path = load_model_path_hallway

    os.makedirs(logdir, exist_ok=True)

    wandb.init(
        project="conformal_rl_prediction",
        mode="online" if not offline else "offline"
    )

    hdim = 256
    future_horizon = max_pred_len
    num_segments = 1
    params = get_params(traj_input_dim, num_intent)
    learning_rate = 1e-4
    max_epochs = 30
    output_len = future_horizon
    diff_order = 1
    hidden_size = hdim
    delta = 0.01
    epsilons = get_knowno_epsilon_values()[:5]
    alpha0s = np.arange(0.01, 0.25, 0.001)[:5]
    alpha0s_simpleset = np.linspace(0.01, 0.6, len(alpha0s))[:5]
    alpha1s = np.arange(0.04, 1, 0.004)[:5]
    temperatures = np.arange(0, 1.801, 0.1)
    debug = True
    train_max_in_set = train_set_size

    train_ds = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=True,
                                       max_pred=future_horizon, debug=debug, min_len=traj_len,
                                       use_habitat=use_habitat, use_vlm=use_vlm, seed=seed,
                                       max_in_set=train_max_in_set)
    test_ds = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=False,
                                      max_pred=future_horizon,  debug=debug, min_len=traj_len,
                                      use_habitat=use_habitat, use_vlm=use_vlm, seed=seed,
                                      max_in_set=train_max_in_set)
    cal_ds = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=False,
                                     max_pred=future_horizon, debug=debug, min_len=traj_len,
                                     max_in_set=calibration_set_size, use_habitat=use_habitat, use_vlm=use_vlm,
                                     is_calibration=True)
    cal_ds_test = IntentPredictionDataset(csv_dir, train_set_size=train_set_size, is_train=False,
                                     max_pred=future_horizon, debug=debug, min_len=traj_len,
                                     max_in_set=50, use_habitat=use_habitat, use_vlm=use_vlm,
                                     calibration_offset=calibration_set_size, is_calibration_test=True)

    if use_vlm:
        collate_dict = collate_fn_stack_only
    else:
        collate_dict = collate_fn

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_dict)
    test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=True, collate_fn=collate_dict)
    cal_loader = DataLoader(cal_ds, batch_size=1, shuffle=True, collate_fn=collate_dict)
    cal_loader_test = DataLoader(cal_ds_test, batch_size=1, shuffle=True, collate_fn=collate_dict)
    num_cal = cal_ds.__len__()

    # TODO: CHANGE BACK
    num_thresh = 10
    lambda_interval = 1 / num_thresh
    lambda_values = np.arange(0, 1, lambda_interval)
    if use_vlm:
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
                                epsilons,
                                delta,
                                alpha0s,
                                alpha0s_simpleset,
                                alpha1s,
                                data_dict,
                                logdir,
                                0)

        knowno_calibration_thresholds = risk_metrics["knowno_calibration_thresholds"]
        calibration_thresholds = risk_metrics["calibration_thresholds"]

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
                       test_cal=True)

        wandb.log(data_dict)
        return


    my_model = IntentFormer(hdim, num_segments, future_horizon, params=params).cuda()
    if load_model_path is not None:
        my_model.load_state_dict(torch.load(load_model_path))
    # my_model = TransformerModel(len(input_cols), input_length, output_length=output_len)
    optimizer = optim.Adam(my_model.parameters(), lr=1e-4)
    # risk_evaluator = RiskEvaluator(self.intent, self.intent_predictor, self.threshold_values,
    #               self.epsilon_values, self.threshold_values_knowno,
    #               self.predict_interval, self.time_limit)



    def lr_lambda(epoch):
        decay_fracs = [1, 0.5, 0.25, 0.125, 0.125/2]
        epoch_drops = [0, 50, 80, 90, 100]
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

    for epoch in range(num_epochs):
        data_dict = {}
        my_model.train()

        if epoch % calibration_interval == 0:

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
                                              epsilons,
                                              delta,
                                              alpha0s,
                                              alpha0s_simpleset,
                                              alpha1s,
                                              data_dict,
                                              logdir,
                                              epoch)

            knowno_calibration_thresholds = risk_metrics["knowno_calibration_thresholds"]
            calibration_thresholds = risk_metrics["calibration_thresholds"]

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
                            test_cal=True)


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


