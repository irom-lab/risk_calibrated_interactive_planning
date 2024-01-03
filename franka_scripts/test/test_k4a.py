
import pyk4a
from pyk4a import PyK4A, Config, ImageFormat
import cv2
import numpy as np
from typing import Optional, Tuple
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d  # noqa: F401


def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
        # h, w = color_image.shape[0:2]
        # h = h // 3 * 2
        # luminance = color_image[:h]
        # chroma = color_image[h:, :w//2]
        # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


# Load camera with the default config
# k4a = PyK4A()
k4a = PyK4A(
    Config(
        # color_resolution=pyk4a.ColorResolution.OFF,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        # synchronized_images_only=False,
        color_resolution=pyk4a.ColorResolution.RES_720P,
        camera_fps=pyk4a.FPS.FPS_5,
        # depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
        synchronized_images_only=True,
    )
)
k4a.start()


# getters and setters directly get and set on device
k4a.whitebalance = 4500
assert k4a.whitebalance == 4500
k4a.whitebalance = 4510
assert k4a.whitebalance == 4510

# # Get the next capture (blocking function)
# capture = k4a.get_capture()
# img_color = capture.color

# # Display with pyplot
# from matplotlib import pyplot as plt
# plt.imshow(img_color[:, :, 2::-1]) # BGRA to RGB
# plt.show()

# while True:
# capture = k4a.get_capture()
# if np.any(capture.depth):
#     cv2.imshow("k4a", colorize(capture.depth, (None, 5000), cv2.COLORMAP_HSV))
#         # key = cv2.waitKey(10)
#         # if key != -1:
#         #     cv2.destroyAllWindows()
#         #     break
# k4a.stop()

while True:
    capture = k4a.get_capture()
    if np.any(capture.depth) and np.any(capture.color):
        break
while True:
    capture = k4a.get_capture()
    if np.any(capture.depth) and np.any(capture.color):
        break
points = capture.depth_point_cloud.reshape((-1, 3))
colors = capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3))

ind = np.where((points[:,2] < 700) & (points[:,2] > 600) & (abs(points[:,0]) < 200) & (abs(points[:,1]) < 200))[0]
ind = np.random.choice(ind, size=5000, replace=False)

points = points[ind]
# print(ind)
# exit()

# print(np.max(points[:,2], axis=0))
# print(np.min(points[:,2], axis=0))
# exit()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    # points[:, 0], points[:, 1], points[:, 2], s=1, c=colors / 255,
    points[:, 0], points[:, 1], points[:, 2], s=1,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
# ax.set_xlim(-2000, 2000)
# ax.set_ylim(-2000, 2000)
# ax.set_zlim(0, 4000)
ax.view_init(elev=-90, azim=-90)
plt.show()

k4a.stop()


