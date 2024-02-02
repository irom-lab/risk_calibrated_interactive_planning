import os
from PIL import Image

def process():

    video_dir = 'video_2'
    hallway_snapshots_dir = '/home/jlidard/PredictiveRL/hallway_snapshots'
    num_img_to_layer = 5
    time_interval = 25
    img_path = os.path.join(hallway_snapshots_dir, video_dir, f'rollout_time0.png')
    background = Image.open(img_path)

    for k in range(1, num_img_to_layer):
        rollout_time = k * time_interval
        img_path = os.path.join(hallway_snapshots_dir, video_dir, f'rollout_time{rollout_time}.png')
        foreground = Image.open(img_path)
        foreground = foreground.convert("RGBA")
        background = background.convert("RGBA")
        new_img = Image.blend(background, foreground, 2*1/(2*k+1))
        background = new_img

    background.save(os.path.join(hallway_snapshots_dir, video_dir, f'composite_{video_dir}.png'))

if __name__ == '__main__':
    process()
