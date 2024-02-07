import pickle
from typing import List
import numpy as np
from scripts.calibrate_hallway import plot_prediction_set_size_versus_success

def load_results():

    vlm_file = '/home/jlidard/PredictiveRL/results/ablation_vlm_1706920710802523929/results_ablation.pkl'
    habitat_file = '/home/jlidard/PredictiveRL/results/ablation_habitat_1706915711299918241/results_ablation.pkl'
    hallway_file = '/home/jlidard/PredictiveRL/results/ablation_hallway_1706911275184028917/results_ablation.pkl'
    plot_again = True

    with open(hallway_file, 'rb') as f:
        data = pickle.load(f)

    agg_prediction_set_size = data["agg_prediction_set_size"]
    agg_task_success_rate = data["agg_task_success_rate"]
    agg_help_rate = data["agg_help_rate"]
    knowno_agg_prediction_set_size = data["knowno_agg_prediction_set_size"]
    knowno_agg_task_success_rate = data["knowno_agg_task_success_rate"]
    knowno_agg_help_rate = data["knowno_agg_help_rate"]
    simple_set_agg_prediction_set_size = data["simple_set_agg_prediction_set_size"]
    simple_set_agg_task_success_rate = data["simple_set_task_success_rate"]
    simple_set_agg_help_rate = data["simple_set_agg_help_rate"]
    entropy_set_agg_prediction_set_size = data["entropy_set_agg_prediction_set_size"]
    entropy_set_agg_task_success_rate = data["entropy_set_agg_task_success_rate"]
    entropy_set_agg_help_rate = data["entropy_set_agg_help_rate"]
    no_help_agg_task_success_rate = data["no_help_agg_task_success_rate"]


    if plot_again:
        img_coverage = plot_prediction_set_size_versus_success(
            agg_prediction_set_size, agg_task_success_rate, agg_help_rate,
            knowno_agg_prediction_set_size, knowno_agg_task_success_rate, knowno_agg_help_rate,
            simple_set_agg_prediction_set_size, simple_set_agg_task_success_rate, simple_set_agg_help_rate,
            entropy_set_agg_prediction_set_size, entropy_set_agg_task_success_rate, entropy_set_agg_help_rate,
            no_help_agg_task_success_rate
        )
        img_coverage.save('/home/jlidard/PredictiveRL/results/ablation_hallway_1706911275184028917/plot_redo.png')

    time_span = len(data["agg_task_success_rate"])
    for i in range(time_span):
        print(f"time {i}")
        for k, v in data.items():
            if type(v) is np.ndarray:
                print(k + ": " + str(v[i]))
            else:
                print(k + ": " + str(v))

if __name__ == '__main__':
    load_results()