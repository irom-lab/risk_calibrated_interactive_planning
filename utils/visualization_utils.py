import matplotlib.pyplot as plt
import numpy as np
import io

from utils.intent_dataset import IntentPredictionDataset

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

