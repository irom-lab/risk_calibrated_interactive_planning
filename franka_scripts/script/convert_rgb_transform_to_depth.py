"""
calibration is done in rgb_camera_link with position (X,Y,Z) and quaternion (X,Y,Z,W) 0.78478 0.04039 0.80117 -0.689 -0.724 0.0006 0.0114. We need to then transform to the depth_camera_link by applying the fixed transformation obtained with rosrun tf tf_echo rgb_camera_link depth_camera_link:
- Translation: [-0.032, -0.002, 0.004]
- Rotation: in Quaternion [-0.051, 0.000, 0.004, 0.999]
            in RPY (radian) [-0.103, 0.001, 0.007]
            in RPY (degree) [-5.900, 0.046, 0.419]
"""
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix, RollPitchYaw, Quaternion

# base_to_rgb_xyz = [0.804, 0.054, 0.780]
# base_to_rgb_xyzw = [-0.7063, -0.7077, 0.0129, 0.009] # xyzw
# base_to_rgb_xyz = [0.782300, 0.041192, 0.812919]
# base_to_rgb_xyzw = [0.706731, 0.707453, 0.005790, -0.002726]
base_to_rgb_xyz = [0.785888, 0.028270, 0.838663]
base_to_rgb_xyzw = [ 0.714781, 0.699075, 0.017054, -0.009583]
base_to_rgb_wxyz = [base_to_rgb_xyzw[-1]]+base_to_rgb_xyzw[:3]
base_to_rgb_wxyz_norm = np.array(base_to_rgb_wxyz) / np.linalg.norm(base_to_rgb_wxyz)

# fixed transform from rgb_camera_link to depth_camera_link
rgb_to_depth_xyz =  [-0.032, -0.002, 0.004] # these are fixed, do not change
rgb_to_depth_xyzw = [-0.051, 0.000, 0.004, 0.999]
rgb_to_depth_wxyz = [rgb_to_depth_xyzw[-1]]+rgb_to_depth_xyzw[:3]
rgb_to_depth_wxyz_norm = np.array(rgb_to_depth_wxyz) / np.linalg.norm(rgb_to_depth_wxyz)

base_to_rgb = RigidTransform(
    RotationMatrix(Quaternion(base_to_rgb_wxyz_norm)), 
    base_to_rgb_xyz)  # eigen uses wxyz
rgb_to_depth = RigidTransform(
    RotationMatrix(Quaternion(rgb_to_depth_wxyz_norm)),
    rgb_to_depth_xyz)

base_to_depth = base_to_rgb.multiply(rgb_to_depth)
print('Translation from base to depth: ', base_to_depth.translation())
print('Rotation (wxyz) from base to depth: ', base_to_depth.rotation().ToQuaternion())

