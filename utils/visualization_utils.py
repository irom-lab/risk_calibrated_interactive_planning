import matplotlib.pyplot as plt
import numpy as np
import io
from matplotlib.transforms import Affine2D
from scipy.stats import multivariate_normal

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1
from utils.intent_dataset import IntentPredictionDataset

import matplotlib

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
        plt.scatter(y_pred[batch, mode, :, 0], y_pred[batch, mode, :, 1], alpha=z_pred[batch, mode].item(), c='b')
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
    img = zz.reshape((xres, yres))
    ax = plt.gca()
    im = ax.imshow(img, cmap='viridis', extent=(0, xmax, 0, ymax), origin='lower', aspect='auto')

    plt.grid()
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cbar, cax = add_colorbar(im, aspect=5)
    # forceAspect(ax,aspect=1)

    cax.set_ylabel(r"Size of Valid Parameter Set")
    plt.xlabel("Help Rate Bound")
    plt.ylabel("Miscoverage Rate Bound")



    # plt.title("Multiple 2D Gaussian Distributions Heatmap")

    return ax

if __name__ == "__main__":

    font = {
            # 'weight': 'bold',
            'size': 16}

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

