import os
from PIL import Image
import cv2
import numpy as np

def alphaMerge(small_foreground, background, top, left):
    """
    Puts a small BGRA picture in front of a larger BGR background.
    :param small_foreground: The overlay image. Must have 4 channels.
    :param background: The background. Must have 3 channels.
    :param top: Y position where to put the overlay.
    :param left: X position where to put the overlay.
    :return: a copy of the background with the overlay added.
    """
    result = background.copy()
    # From everything I read so far, it seems we need the alpha channel separately
    # so let's split the overlay image into its individual channels
    fg_b, fg_g, fg_r, fg_a = cv2.split(small_foreground)
    # Make the range 0...1 instead of 0...255
    fg_a = fg_a / 255.0
    # Multiply the RGB channels with the alpha channel
    label_rgb = cv2.merge([fg_b * fg_a, fg_g * fg_a, fg_r * fg_a])

    # Work on a part of the background only
    height, width = small_foreground.shape[0], small_foreground.shape[1]
    part_of_bg = result[top:top + height, left:left + width, :]
    # Same procedure as before: split the individual channels
    bg_b, bg_g, bg_r = cv2.split(part_of_bg)
    # Merge them back with opposite of the alpha channel
    part_of_bg = cv2.merge([bg_b * (1 - fg_a), bg_g * (1 - fg_a), bg_r * (1 - fg_a)])

    # Add the label and the part of the background
    cv2.add(label_rgb, part_of_bg, part_of_bg)
    # Replace a part of the background
    result[top:top + height, left:left + width, :] = part_of_bg
    return result



def process():

    video_dir = 'video_3'
    hallway_snapshots_dir = '/home/jlidard/PredictiveRL/hallway_snapshots'
    num_img_to_layer = 5
    time_interval = 15
    start_time = 20
    img_path = os.path.join(hallway_snapshots_dir, video_dir, f'rollout_time{start_time}.png')
    background = Image.open(img_path)

    for k in range(1, num_img_to_layer):
        rollout_time = k * time_interval + start_time
        img_path = os.path.join(hallway_snapshots_dir, video_dir, f'rollout_time{rollout_time}.png')
        foreground = Image.open(img_path)
        foreground = foreground.convert("RGBA")
        background = background.convert("RGBA")
        new_img = Image.blend(background, foreground, 2/(0.75*k+1))
        background = new_img

    background.save(os.path.join(hallway_snapshots_dir, video_dir, f'composite_{video_dir}.png'))

if __name__ == '__main__':
    process()
