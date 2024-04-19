import pickle
from typing import List
import numpy as np
from utils.visualization_utils import plot_prediction_set_size_versus_success
from utils.general_utils import linear_interpolation
import os
from prettytable import PrettyTable

def load_results():



    vlm_file = '/home/jlidard/risk_calibrated_interactive_planning/results/ablation_vlm_1713137550712558280/results_ablation.pkl'
    habitat_file = '/home/jlidard/risk_calibrated_interactive_planning/results/ablation_habitat_1713142001886709833/results_ablation.pkl'
    hallway_file = '/home/jlidard/risk_calibrated_interactive_planning/results/ablation_hallway_1713142198427181366/results_ablation.pkl'
    bimanual_file = '/home/jlidard/risk_calibrated_interactive_planning/results/ablation_bimanual_1713282708717469269/results_ablation.pkl'

    hallway_early_stopping_1 = '/home/jlidard/risk_calibrated_interactive_planning/results/earlystopping_ablation_hallway_1713080466634831396/results_ablation.pkl'
    hallway_early_stopping_2 = '/home/jlidard/risk_calibrated_interactive_planning/results/earlystopping_ablation_hallway_1713080680738148024/results_ablation.pkl'
    hallway_early_stopping_3 = '/home/jlidard/risk_calibrated_interactive_planning/results/earlystoppingablation_hallway_1713080871082080957/results_ablation.pkl'

    plot_again = True
    exps = [hallway_file, habitat_file, vlm_file, bimanual_file, hallway_early_stopping_1, hallway_early_stopping_2, hallway_early_stopping_3]
    labels = ["hallway", "habitat", "vlm", "bimanual", "early stopping 1", "early stopping 2", "early stopping 3"]
    for exp, label in zip(exps, labels):
        with open(exp, 'rb') as f:
            data = pickle.load(f)

        agg_prediction_set_size = data["agg_prediction_set_size"]
        agg_task_success_rate = data["agg_task_success_rate"]
        agg_help_rate = data["agg_help_rate"]
        agg_task_success_rate_stage = data["agg_task_success_rate_stage"]
        agg_help_rate_stage = data["agg_help_rate_stage"]
        knowno_agg_prediction_set_size = data["knowno_agg_prediction_set_size"]
        knowno_agg_task_success_rate = data["knowno_agg_task_success_rate"]
        knowno_agg_help_rate = data["knowno_agg_help_rate"]
        knowno_agg_task_success_rate_stage = data["knowno_agg_task_success_rate_stage"]
        knowno_agg_help_rate_stage = data["knowno_agg_help_rate_stage"]
        simple_set_agg_prediction_set_size = data["simple_set_agg_prediction_set_size"]
        simple_set_agg_task_success_rate = data["simple_set_task_success_rate"]
        simple_set_agg_help_rate = data["simple_set_agg_help_rate"]
        simple_set_agg_task_success_rate_stage = data["simple_set_task_success_rate_stage"]
        simple_set_agg_help_rate_stage = data["simple_set_agg_help_rate_stage"]
        entropy_set_agg_prediction_set_size = data["entropy_set_agg_prediction_set_size"]
        entropy_set_agg_task_success_rate = data["entropy_set_agg_task_success_rate"]
        entropy_set_agg_help_rate = data["entropy_set_agg_help_rate"]
        entropy_set_agg_task_success_rate_stage = data["entropy_set_agg_task_success_rate_stage"]
        entropy_set_agg_help_rate_stage = data["entropy_set_agg_help_rate_stage"]
        no_help_agg_task_success_rate = data["no_help_agg_task_success_rate"]
        no_help_agg_task_success_rate_stage = data["no_help_agg_task_success_rate_stage"]

        grid_points = len(simple_set_agg_task_success_rate)
        alpha0s_simpleset = np.linspace(0, 1, grid_points)


        if plot_again:
            file_path = exp.split('/')
            new_plot_name = 'plot_redo.png'
            new_filepath = os.path.join('/home/', *file_path[2:-1], new_plot_name)
            target = 0.5
            zero_crossings = -1 # np.where(np.diff(np.sign(agg_task_success_rate - target)))[0][-1]

            img_coverage = plot_prediction_set_size_versus_success(
                agg_prediction_set_size[:zero_crossings],
                agg_task_success_rate[:zero_crossings],
                agg_help_rate[:zero_crossings],
                knowno_agg_prediction_set_size,
                knowno_agg_task_success_rate,
                knowno_agg_help_rate,
                simple_set_agg_prediction_set_size,
                simple_set_agg_task_success_rate,
                simple_set_agg_help_rate,
                entropy_set_agg_prediction_set_size,
                entropy_set_agg_task_success_rate,
                entropy_set_agg_help_rate,
                no_help_agg_task_success_rate,
                calibration_temps=None,
            )
            img_coverage.save(new_filepath)

        target_val = 0.75 #if not label == "vlm" else 0.8
        plan_result, help_result, success_stage_result, help_stage_result = get_intermediates_lin_interp(
            agg_task_success_rate, agg_help_rate, agg_task_success_rate_stage, agg_help_rate_stage,
            target_val=target_val,
            #interp_on=exp not in exps[3:]
        )
        knowno_plan_result, knowno_help_result, knowno_success_stage_result, knowno_help_stage_result = get_intermediates_lin_interp(
            knowno_agg_task_success_rate, knowno_agg_help_rate,
            knowno_agg_task_success_rate_stage, knowno_agg_help_rate_stage,
            target_val=target_val,
            # interp_on=exp not in exps[3:]
        )
        simple_set_plan_result, simple_set_help_result, simple_set_success_stage_result, simple_set_help_stage_result =\
            get_intermediates_lin_interp(
            simple_set_agg_task_success_rate[::-1], simple_set_agg_help_rate[::-1],
            simple_set_agg_task_success_rate_stage[::-1], simple_set_agg_help_rate_stage[::-1],
            target_val=target_val,
            alphas=alpha0s_simpleset,
            # interp_on=exp not in exps[3:]
        )
        table = PrettyTable()
        table.field_names = ["Method", "Plan Success Rate", "Help Rate", "Step Sucess Rate", "Step Help"]
        table.add_rows(
            [
                ["RCIP", plan_result, help_result, success_stage_result, help_stage_result],
                ["KnowNo", knowno_plan_result, knowno_help_result, knowno_success_stage_result, knowno_help_stage_result],
                ["Simple Set", simple_set_plan_result, simple_set_help_result, simple_set_success_stage_result, simple_set_help_stage_result],
                ["Entropy Set", entropy_set_agg_task_success_rate, entropy_set_agg_help_rate, entropy_set_agg_task_success_rate_stage, entropy_set_agg_help_rate_stage],
                ["No Help", no_help_agg_task_success_rate[0], 0, no_help_agg_task_success_rate_stage[0], 0],
            ]
        )
        print(label)
        print(table)
        print()


def get_intermediates_lin_interp(agg_task_success_rate, agg_help_rate, agg_task_success_rate_stage, agg_help_rate_stage,
                                 target_val=0.85, interp_on=False, alphas=None):
    try:
        i_under, i_over, success_under, success_over = get_closest_index_success(agg_task_success_rate, target_val)
    except:
        i_under, i_over, success_under, success_over = (0, 0, 1, 1)

    plan_result = success_over

    if alphas is not None:
        print(1-alphas[i_under], 1-alphas[i_over])

    # Help
    help_under = agg_help_rate[i_under]
    help_over = agg_help_rate[i_over]
    x_y_pairs = [(help_under, success_under), (help_over, success_over)]
    help_result = help_over #
    if interp_on:
        help_result = linear_interpolation(x_y_pairs, target_val)

    # Success Stage
    success_stage_under = agg_task_success_rate_stage[i_under]
    success_stage_over = agg_task_success_rate_stage[i_over]
    x_y_pairs = [(success_stage_under, success_under), (success_stage_under, success_over)]
    success_stage_result = success_stage_over # linear_interpolation(x_y_pairs, target_val)
    if interp_on:
        success_stage_result = linear_interpolation(x_y_pairs, target_val)

    # Help Stage
    help_stage_under = agg_help_rate_stage[i_under]
    help_stage_over = agg_help_rate[i_over]
    x_y_pairs = [(help_stage_under, success_under), (help_stage_over, success_over)]
    help_stage_result = help_stage_over # linear_interpolation(x_y_pairs, target_val)
    if interp_on:
        help_stage_result = linear_interpolation(x_y_pairs, target_val)

    return plan_result, help_result, success_stage_result, help_stage_result


def get_closest_index_success(success_rates, target=0.85):
    zero_crossings = np.where(np.diff(np.sign(success_rates-target)))[0][-1]
    # find the closest lambda to knowno thresh, without going over
    under = success_rates[zero_crossings+1]
    over = success_rates[zero_crossings-0]

    i_under = zero_crossings + 1
    i_over = zero_crossings - 0

    return i_under, i_over, under, over

if __name__ == '__main__':
    load_results()