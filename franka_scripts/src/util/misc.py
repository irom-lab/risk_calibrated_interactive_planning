import numpy as np
import torch
import time


def rotate_tensor(orig_tensor, theta):
	"""
	Rotate images clockwise
	"""
	affine_mat = np.array([[np.cos(theta), np.sin(theta), 0],
							[-np.sin(theta), np.cos(theta), 0]])
	affine_mat.shape = (2,3,1)
	affine_mat = torch.from_numpy(affine_mat).permute(2,0,1).float()
	flow_grid = torch.nn.functional.affine_grid(affine_mat, orig_tensor.size(), align_corners=False)
	return torch.nn.functional.grid_sample(orig_tensor, flow_grid, mode='nearest', align_corners=False)


def wrap2halfPi(angle):  # assume input in [-pi, pi]
    if angle < -np.pi/2:
        return angle + np.pi 
    elif angle > np.pi/2:
        return angle - np.pi
    return angle


def wrap2pi(angle):
    if angle < -np.pi:
        return angle + 2*np.pi 
    elif angle > np.pi:
        return angle - 2*np.pi
    return angle


def bin_image(image_raw, target_height, target_width, bin_average=True):
    """
    Assume square image out
    """
    image_out = np.zeros((target_height, target_width))
    raw_height, raw_width = image_raw.shape

    start_time = time.time()
    for height_ind in range(target_height):
        for width_ind in range(target_width):
            # find the top left pixel involving in raw image
            first_height_pixel = np.floor(height_ind/target_height*raw_height).astype('int')
            end_height_pixel = np.ceil(height_ind/target_height*raw_height).astype('int')+1
            first_width_pixel = np.floor(width_ind/target_width*raw_width).astype('int')
            end_width_pixel = np.ceil(width_ind/target_width*raw_width).astype('int')+1

            if bin_average:
                image_out[height_ind, width_ind] = np.mean(image_raw[first_height_pixel:end_height_pixel, \
                    first_width_pixel:end_width_pixel])
            else:  # use max
                image_out[height_ind, width_ind] = np.max(image_raw[first_height_pixel:end_height_pixel, \
                    first_width_pixel:end_width_pixel])
    # print('Time used:', time.time()-start_time)
    return image_out
