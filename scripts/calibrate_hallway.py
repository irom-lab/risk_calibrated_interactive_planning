import numpy as np
from stable_baselines3 import PPO, SAC  # , MultiModalPPO
# from sb3_contrib import RecurrentPPO
from environments.hallway_env import HallwayEnv
from environments.make_vectorized_hallway_env import make_env, make_bullet_env
import os
import time
import pandas as pd
from PIL import Image
from environments.hallway_env import HumanIntent
import matplotlib.pyplot as plt
from environments.hallway_env import HallwayEnv, prompt
from environments.pybullet_hallway_env import BulletHallwayEnv, prompt
from os.path import expanduser
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook
from scipy.stats import binom, beta
from utils.visualization_utils import get_img_from_fig, add_colorbar
import torch
import matplotlib
from matplotlib.ticker import FormatStrFormatter

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1


trigger_sync = TriggerWandbSyncHook()  # <--- New!

home = expanduser("~")

use_bullet = True
if not use_bullet:
    model_num = 1700717235  # Best RGB
    model_num = 1701999480  # Best kinematic
else:
    model_num = 1702322915 # 300 from 3d hallway

loaddir = os.path.join(home, f"PredictiveRL/models/{model_num}/epoch_100.zip")
logdir = os.path.join(home, f"PredictiveRL/conformal_outputs/{int(time.time())}/")
dataframe_path = os.path.join(home, f"PredictiveRL/conformal_outputs/{int(time.time())}.csv")


def plot_figures(non_conformity_score, qhat, is_bullet=False, save_fig=False):

    # plot histogram and quantile
    plt.figure()
    plt.figure(figsize=(6, 2))
    plt.hist(non_conformity_score, bins=30, edgecolor='k', linewidth=1)
    plt.axvline(
        x=qhat, linestyle='--', color='r', label='Quantile value'
    )
    plt.title(
        'Histogram of non-comformity scores in the calibration set'
    )
    plt.xlim((0, 1))
    plt.xlabel('Non-comformity score')
    plt.legend()
    if not is_bullet:
        name = 'hallway_non_conformity.png'
    else:
        name = 'bullet_hallway_non_conformity.png'
    if save_fig:
        plt.savefig(name)

    img = get_img_from_fig(plt.gcf())
    plt.close('all')
    return img


def plot_nonsingleton_figure(lambdas, prediction_set_size, alpha=0.15, is_bullet=False, save_fig=False):

    # plot histogram and quantile
    x_list = lambdas
    y_list = prediction_set_size
    plt.figure(figsize=(6, 2))
    plt.scatter(x_list, y_list, edgecolor='k', linewidth=1)
    plt.axhline(
        y=alpha, linestyle='--', color='r', label='Limit Value'
    )
    plt.title(
        'Nonsingleton Rate versus Prediction Set Threshold'
    )
    plt.xlabel('Prediction Set Threshold')
    plt.legend()
    plt.axis((0, 1, 0, 0.6))
    if not is_bullet:
        name = 'hallway_set_size.png'
    else:
        name = 'bullet_hallway_set_size.png'

    if save_fig:
        plt.savefig(name)

    img = get_img_from_fig(plt.gcf())
    plt.close('all')
    return img

def plot_miscoverage_figure(lambdas, miscoverage_rate, alpha=0.15, is_bullet=False, save_fig=False):

    # plot histogram and quantile
    x_list = lambdas
    y_list = miscoverage_rate
    plt.figure(figsize=(6, 2))
    plt.scatter(x_list, y_list, edgecolor='k', linewidth=1)
    plt.axhline(
        y=alpha, linestyle='--', color='r', label='Limit Value'
    )
    plt.title(
        'Miscoverage Rate versus Prediction Threshold'
    )
    plt.xlabel('Threshold')
    plt.axis((0, 1, 0, 0.25))
    plt.legend()

    if not is_bullet:
        name = 'hallway_set_size.png'
    else:
        name = 'bullet_hallway_miscoverage_ragte.png'

    if save_fig:
        plt.savefig(name)


    img = get_img_from_fig(plt.gcf())
    plt.close('all')
    return img




# Adjusting the function to remove the invalid property 'shrink'

def label_with_arrow(ax, text, xy, xytext, arrowprops):
    """
    Adds a label with an arrow to the plot.
    Args:
    - ax: The axis object to add the label to.
    - text (str): The text of the label.
    - xy (tuple): The point (x, y) to point the arrow to.
    - xytext (tuple): The position (x, y) to place the text at.
    - arrowprops (dict): Properties for the arrow that points from the text to the point.


    """

    # fig_xy = ax.figure.transFigure.transform_point(xy)
    # # Transform the display coordinates to data coordinates
    # inv = ax.transData.inverted()
    # data_xy = inv.transform_point(fig_xy)
    #
    # fig_xytext = ax.figure.transFigure.transform_point(xytext)
    # # Transform the display coordinates to data coordinates
    # inv = ax.transData.inverted()
    # data_xytext = inv.transform_point(fig_xytext)


    ax.annotate(text, xy=xy, xytext=xytext, arrowprops=arrowprops, xycoords='subfigure fraction', textcoords='subfigure fraction')


def plot_prediction_set_size_versus_success(prediction_set_size, task_success_rate, help_rate,
                                            knowno_prediction_set_size, knowno_task_success_rate, knowno_help_rate,
                                            simple_set_prediction_set_size, simple_set_task_success_rate, simple_set_help_rate,
                                            entropy_set_prediction_set_size, entropy_set_task_success_rate, entropy_set_help_rate,
                                            no_help_task_success_rate, save_fig=False):

    # plot histogram and quantile
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(32, 9))
    ax1.plot(prediction_set_size, task_success_rate, label='RCIP', linewidth=3)
    ax1.plot(knowno_prediction_set_size, knowno_task_success_rate, label='KnowNo', linewidth=3)
    ax1.plot(simple_set_prediction_set_size, simple_set_task_success_rate, label='Simple Set', linewidth=3)
    ax1.plot(entropy_set_prediction_set_size, entropy_set_task_success_rate, label='Entropy Set', linewidth=3)
    ax1.set_ylabel('Task Success Rate')
    ax1.set_xlabel('Prediction Set Size')
    ax1.set_xlim([1, 5])
    ax1.set_ylim([.6, 1])
    major_ticks_x = [1, 2, 3, 4, 5]
    major_ticks_y = [0.6, 0.7, 0.8, 0.9, 1.0]
    ax1.set_xticks(major_ticks_x)
    ax1.set_yticks(major_ticks_y)
    label_with_arrow(ax1, r'lower $\alpha_2$', xy=(0.42, 0.5), xytext=(0.35, 0.3),
                     arrowprops=dict(facecolor='black', arrowstyle='->', lw=3, linestyle='dashed'))

    # label_with_arrow(ax1, 'lower ε', xy=(1, 0.9), xytext=(0.95, 0.8),
    #                  arrowprops=dict(facecolor='black', arrowstyle='->'))

    ax2.plot(prediction_set_size, task_success_rate, linewidth=3)
    ax2.plot(knowno_prediction_set_size, knowno_task_success_rate, linewidth=3)
    ax2.plot(simple_set_prediction_set_size, simple_set_task_success_rate, linewidth=3)
    ax2.plot(entropy_set_prediction_set_size, entropy_set_task_success_rate, linewidth=3)
    ax2.scatter(np.zeros_like(no_help_task_success_rate), no_help_task_success_rate, s=500, marker="*", label="No help")
    ax2.set_ylabel('Task Success Rate')
    ax2.set_xlabel('Human Help Rate')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([.6, 1])
    major_ticks_x = [0, 0.2, 0.4, 0.6, 0.8, 1]
    major_ticks_y = [0.6, 0.7, 0.8, 0.9, 1.0]
    ax2.set_xticks(major_ticks_x)
    ax2.set_yticks(major_ticks_y)
    labels = [item.get_text() for item in ax2.get_xticklabels()]
    labels[0] = "0"
    labels[-1] = "1"
    # Beat them into submission and set them back again
    ax2.set_xticklabels(labels)
    label_with_arrow(ax2, r'lower $\alpha_1$', xy=(0.92, 0.5), xytext=(0.85, 0.3),
                     arrowprops=dict(facecolor='black', arrowstyle='->', lw=3, linestyle='dashed'))

    f.legend(loc="lower center", ncol=5, frameon=False)

    # Adjust the layout to make room for the shared legend
    f.tight_layout(rect=[0, 0.1, 1, 1])

    save_to = '/home/jlidard/PredictiveRL/figures/prediction_set_size_versus_task_success.png'

    if save_fig:
        plt.savefig(save_to)

    return get_img_from_fig(plt.gcf())


def plot_prediction_success_versus_help_bound(task_success_rate, help_rate_bound, temperature, save_fig=False):

    # plot histogram and quantile
    f, ax1 = plt.subplots(ncols=1, figsize=(16, 9))
    sc = ax1.scatter(help_rate_bound, task_success_rate, s=400, c=temperature, cmap="turbo", vmin=temperature.min(),
                vmax=temperature.max())
    ax1.set_xlabel('Help Rate Bound')
    ax1.set_ylabel('RCIP Parameter Set Size')
    ax1.set_xlim([0, 0.2])
    ax1.set_ylim([0, 10])
    major_ticks_x = [0, 0.2, 0.4, 0.6, 0.8, 1]
    major_ticks_y = [0.6, 0.7, 0.8, 0.9, 1.0]
    ax1.set_xticks(major_ticks_x)
    # ax1.set_yticks(major_ticks_y)
    labels = [item.get_text() for item in ax1.get_xticklabels()]
    labels[0] = "0"
    labels[-1] = "1"
    ax1.set_xticklabels(labels)
    plt.axis('equal')
    cbar, cax = add_colorbar(sc, aspect=20, pad_fraction=1.0)
    cax.set_title(r'$\tau$=+1.0')
    cax.set_xlabel(r'$\tau$=0.0')
    cbar.set_ticks([])

    plt.show()


    save_to = '/home/jlidard/PredictiveRL/figures/prediction_set_size_versus_task_success.png'

    if save_fig:
        plt.savefig(save_to)

    return get_img_from_fig(plt.gcf())


def hoeffding_bentkus(risk_values, alpha_val=0.9, n=100):
    sample_risk_mean = risk_values.mean(-1)
    alpha_val = torch.Tensor([alpha_val]).to(risk_values.device)

    max_alpha = torch.minimum(risk_values, alpha_val)
    ce = cross_entropy(max_alpha, alpha_val).clip(min=0)  # clip avoids p-val > 1 (p-val will still be bad)
    left_term = np.exp(-n * ce)

    x = np.ceil(n * sample_risk_mean)
    bin_cdf = binom.cdf(x, n, alpha_val)
    right_term = np.e * bin_cdf
    right_term = torch.Tensor(right_term).to(risk_values.device)

    hb_p_val = torch.minimum(left_term, right_term)
    return hb_p_val

def cross_entropy(a, b):
    return a * np.log(a/(b+0.001)+0.001) + (1-a) * np.log((1-a)/(1-b+0.001) + 0.001)

def knowno_test_eval(N=500, eps_coverage=0.02, delta=0.01):
    v = np.floor((N + 1)*eps_coverage)
    a = N + 1 - v
    b = v
    return 1-beta.ppf(delta, a, b)


def get_knowno_epsilon_values():

    knowno_coverage_range = np.arange(0.01, .25, 0.001)
    indices = list(range(len(knowno_coverage_range)))
    test_eps_vals = [knowno_test_eval(eps_coverage=e) for e in knowno_coverage_range]
    test_eps = [(i, j, k) for (i,j,k) in zip(indices, knowno_coverage_range, test_eps_vals)]
    return test_eps_vals


if __name__ == "__main__":
    x=knowno_test_eval()
    print(x)

    risk = torch.Tensor([0.03])
    y = hoeffding_bentkus(risk, alpha_val=0.04, n=500)
    print(y.item())



    # Generate random curves
    x = np.linspace(0, 1, 100)
    curves = []
    num_curves = 5

    for _ in range(num_curves):
        # Random coefficients for a polynomial curve
        coefficients = np.random.rand(4)
        polynomial = np.poly1d(coefficients)

        # Generate y values and ensure start and end points are (0,0) and (1,1)
        y = polynomial(x)
        y -= y[0]
        y /= y[-1]
        y *= (1 - y[0])

        curves.append(y)

    font = {
            # 'weight': 'bold',
            'size': 32}

    matplotlib.rc('font', **font)

    # plot_prediction_set_size_versus_success(x, curves[0], curves[0],
    #                                         x, curves[1], curves[1],
    #                                         x, curves[2], curves[2],
    #                                         x, curves[3], curves[3],
    #                                         curves[-1][50], None)

    # plot_prediction_success_versus_help_bound(x, curves[1], x)






