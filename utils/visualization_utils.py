import matplotlib.pyplot as plt
import numpy as np
import io

import torch
from matplotlib.transforms import Affine2D
from scipy.stats import multivariate_normal

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1
from utils.intent_dataset import IntentPredictionDataset

import matplotlib

import numpy as np
from stable_baselines3 import PPO, SAC  # , MultiModalPPO
# from sb3_contrib import RecurrentPPO
from os.path import expanduser
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook
from scipy.stats import binom, beta
import torch
import matplotlib
from matplotlib.ticker import FormatStrFormatter

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1


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
                                            no_help_task_success_rate, calibration_temps, save_fig=False, success_rate_lower_bound=0.5):

    # plot histogram and quantile
    # f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(32, 9))
    # ax1.plot(prediction_set_size, task_success_rate, label='RCIP', linewidth=3)
    # ax1.plot(knowno_prediction_set_size, knowno_task_success_rate, label='KnowNo', linewidth=3)
    # ax1.plot(simple_set_prediction_set_size, simple_set_task_success_rate, label='Simple Set', linewidth=3)
    # # ax1.plot(entropy_set_prediction_set_size, entropy_set_task_success_rate, label='Entropy Set', linewidth=3)
    # ax1.set_ylabel('Task Success Rate')
    # ax1.set_xlabel('Prediction Set Size')
    # ax1.set_xlim([0, 5])
    # ax1.set_ylim([0, 1])
    # major_ticks_x = [1, 2, 3, 4, 5]
    # major_ticks_y = [0.6, 0.7, 0.8, 0.9, 1.0]
    # ax1.set_xticks(major_ticks_x)
    # ax1.set_yticks(major_ticks_y)
    # label_with_arrow(ax1, r'lower $\alpha_2$', xy=(0.42, 0.5), xytext=(0.35, 0.3),
    #                  arrowprops=dict(facecolor='black', arrowstyle='->', lw=3, linestyle='dashed'))

    font = {
            # 'weight': 'bold',
            'size': 26}

    matplotlib.rc('font', **font)

    f, ax2 = plt.subplots(1, figsize=(10, 10))

    # label_with_arrow(ax1, 'lower ε', xy=(1, 0.9), xytext=(0.95, 0.8),
    #                  arrowprops=dict(facecolor='black', arrowstyle='->'))

    ax2.plot(help_rate, task_success_rate, linewidth=5, label="RCIP (ours)", c='C1')
    ax2.plot(knowno_help_rate, knowno_task_success_rate, linewidth=5, label="KnowNo", c='#1f77b4')
    ax2.plot(simple_set_help_rate, simple_set_task_success_rate, linewidth=5, label="Simple Set", c='C2')
    ax2.scatter(entropy_set_help_rate, entropy_set_task_success_rate, s=1000, marker="*", label="Entropy Set", color='m')
    ax2.scatter(np.zeros_like(no_help_task_success_rate), no_help_task_success_rate, s=1000, marker="*", label="No help", color='black')
    # plt.scatter(help_rate, task_success_rate, c=calibration_temps, cmap='magma', s=500, # vmin=min(calibration_temps), vmax=max(calibration_temps),
    #             norm=matplotlib.colors.LogNorm())
    print(calibration_temps)
    # plt.colorbar()
    ax2.set_ylabel('Plan Success Rate')
    ax2.set_xlabel('Human Help Rate')
    ax2.set_xlim([-0.02, 1])
    success_rate_lower_bound = 0.5
    ax2.set_ylim([success_rate_lower_bound, 1.02])
    # plt.axis('equal')
    # major_ticks_x = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # major_ticks_y = [0.6, 0.7, 0.8, 0.9, 1.0]
    # ax2.set_xticks(major_ticks_x)
    # ax2.set_yticks(major_ticks_y)
    # labels = [item.get_text() for item in ax2.get_xticklabels()]
    # labels[0] = "0"
    # labels[-1] = "1"
    # Beat them into submission and set them back again
    # ax2.set_xticklabels(labels)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    label_with_arrow(ax2, r'lower $\alpha_{\text{cov}}$', xy=(0.85, 0.65), xytext=(0.6, 0.45),
                     arrowprops=dict(facecolor='black', arrowstyle='->', lw=3, linestyle='dashed'))
    # label_with_arrow(ax2, r'lower $\alpha_{\text{cov}}$', xy=(0.75, 0.75), xytext=(0.5, 0.55),
    #                  arrowprops=dict(facecolor='black', arrowstyle='->', lw=3, linestyle='dashed'))

    f.legend(loc="lower center", ncol=3, frameon=False)

    # Adjust the layout to make room for the shared legend
    f.tight_layout(rect=[0, 0.1, 1, 1])

    save_to = '/home/jlidard/PredictiveRL/figures/prediction_set_size_versus_task_success.png'

    if save_fig:
        plt.savefig(save_to)

    return get_img_from_fig(plt.gcf())


def plot_prediction_success_versus_help_bound(task_success_rate, help_rate_bound, set_size, save_fig=False):
    font = {
            # 'weight': 'bold',
            'size': 26}

    matplotlib.rc('font', **font)
    # plot histogram and quantile
    f, ax1 = plt.subplots(ncols=1, figsize=(16, 9))
    sc = ax1.scatter(help_rate_bound, task_success_rate, s=400, c=set_size, cmap="plasma", vmin=set_size.min(),
                vmax=set_size.max())
    ax1.set_xlabel('Help Rate Bound')
    ax1.set_ylabel('RCIP Parameter Set Size')
    ax1.set_xlim([0, task_success_rate[-1]])
    ax1.set_ylim([0, help_rate_bound[-1]])
    # major_ticks_x = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # major_ticks_y = [0.6, 0.7, 0.8, 0.9, 1.0]
    # ax1.set_xticks(major_ticks_x)
    # # ax1.set_yticks(major_ticks_y)
    labels = [item.get_text() for item in ax1.get_xticklabels()]
    labels[0] = "0"
    labels[-1] = "1"
    ax1.set_xticklabels(labels)
    plt.axis('equal')
    cbar, cax = add_colorbar(sc, aspect=40, pad_fraction=0.25)
    cax.set_title(rf'$|\hat \Phi|$={max(set_size)}')
    cax.set_xlabel(r'$\|\hat \Phi|$=0')
    cbar.set_ticks([])

    plt.show()


    save_to = '/home/jlidard/PredictiveRL/figures/prediction_set_size_versus_task_success.png'

    if save_fig:
        plt.savefig(save_to)

    return get_img_from_fig(plt.gcf())


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs), cax

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def get_img_from_fig(fig):
    # img = PIL.Image.frombytes('RGB',fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    plt.close(fig)
    io_buf.close()
    img = PIL.Image.fromarray(img)
    return img

import PIL
def plot_pred(batch_X, robot_state_gt, human_state_gt, batch_z, y_pred, z_pred, batch=0, use_habitat=False):
    y_pred = y_pred.cpu()
    robot_state_gt = robot_state_gt.cpu()
    human_state_gt = human_state_gt.cpu()
    z_pred = z_pred.cpu()
    batch_X = batch_X.cpu()

    if use_habitat:
        robot_ind = 1
        human_ind = 4

        fig = plt.figure()

        plt.scatter(batch_X[batch, :, robot_ind], batch_X[batch, :, robot_ind + 2], c='black')
        plt.scatter(batch_X[batch, :, human_ind], batch_X[batch, :, human_ind + 2], c='black')
        plt.scatter(robot_state_gt[batch, :, 0], robot_state_gt[batch, :, 2], c='orange')
        plt.scatter(human_state_gt[batch, :, 0], human_state_gt[batch, :, 2], c='orange')
    else:
        robot_ind = 17
        human_ind = 20

        fig = plt.figure()

        plt.scatter(batch_X[batch, :, robot_ind], batch_X[batch, :, robot_ind+1], c='black')
        plt.scatter(batch_X[batch, :, human_ind], batch_X[batch, :, human_ind+1], c='black')
        plt.scatter(robot_state_gt[batch, :, 0], robot_state_gt[batch, :, 1], c='orange')
        plt.scatter(human_state_gt[batch, :, 0], human_state_gt[batch, :, 1], c='orange')

    y_pred = y_pred.detach().cpu()
    z_pred = z_pred.detach().cpu()

    for mode in range(y_pred.shape[1]):
        alpha = z_pred[batch, mode].item()
        alpha = 1
        plt.scatter(y_pred[batch, mode, :, 0], y_pred[batch, mode, :, 1], alpha=alpha, c='b')
    plt.axis((-6, 6, -4.5, 4.5))

    img = get_img_from_fig(fig)

    return img

def draw_heatmap(alpha1s, alpha2s, parameter_set_sizes, xmax=1, ymax=0.15):
    """
    Generate a heatmap of multiple weighted 2D Gaussian distributions on the same plot.

    Parameters:

    Returns:
    matplotlib.figure.Figure: A heatmap of the weighted Gaussian distributions.
    """

    font = {
            # 'weight': 'bold',
            'size': 26}

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    x = alpha1s
    y = alpha2s
    xres = len(alpha1s)
    yres = len(alpha2s)
    xmax = x[-1]
    ymax = y[-1]
    xx, yy = np.meshgrid(x, y)
    zz = parameter_set_sizes

    # reshape and plot image
    zz = np.array(zz)
    img = zz.reshape((xres, yres))
    ax = plt.gca()
    im = ax.imshow(img, cmap='viridis', extent=(0, xmax, 0, ymax), origin='lower', aspect='auto')

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cbar, cax = add_colorbar(im, aspect=20, pad_fraction=0.5)
    # forceAspect(ax,aspect=1)

    cax.set_ylabel(r"Size of Valid Parameter Set")
    plt.xlabel("Miscoverage Rate Bound")
    plt.ylabel("Help Rate Bound")

    fig = plt.gcf()
    img = get_img_from_fig(fig)

    # plt.title("Multiple 2D Gaussian Distributions Heatmap")

    return img

if __name__ == "__main__":

    font = {
            # 'weight': 'bold',
            'size': 26}

    matplotlib.rc('font', **font)

    zs = np.linspace(0, 10, 240*240)

    alpha1s = np.linspace(0, 1, 240)
    alpha2s = np.linspace(0, 0.25, 240)

    k = 0

    for j, y in enumerate(alpha2s):
        for i, x in enumerate(alpha1s):
            zs[k] = 5 * x + y*20
            k += 1
    draw_heatmap(alpha1s, alpha2s, zs)
    plt.show()

