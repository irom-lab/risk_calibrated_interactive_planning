import matplotlib.pyplot as plt

def get_img_from_fig(fig):
    img = PIL.Image.frombytes('RGB',fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    return img

import PIL
def plot_pred(batch_X, robot_state_gt, human_state_gt, batch_z, y_pred, z_pred, batch=0):
    y_pred = y_pred.cpu()
    robot_state_gt = robot_state_gt.cpu()
    human_state_gt = human_state_gt.cpu()
    z_pred = z_pred.cpu()

    fig = plt.figure()

    plt.scatter(robot_state_gt[batch, :, 0], robot_state_gt[batch, :, 1], c='orange')
    plt.scatter(human_state_gt[batch, :, 0], human_state_gt[batch, :, 1], c='orange')

    y_pred = y_pred.detach().cpu()
    z_pred = z_pred.detach().cpu()

    for mode in range(y_pred.shape[1]):
        plt.scatter(y_pred[batch, mode, :, 0], y_pred[batch, mode, :, 1], alpha=z_pred[batch, mode].item(), c='b')

    img = get_img_from_fig(fig)

    return img

