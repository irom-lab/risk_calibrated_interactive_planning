import numpy as np
import pybullet as p
import pybullet_data
from arm_env import ArmEnv
import time

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)
#time.sleep(5)

p.loadURDF("/Users/justinlidard/bullet3/examples/pybullet/gym/pybullet_data/plane.urdf", [0, 0, 0], [0, 0, 0, 1])
targid = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
obj_of_focus = targid

n = p.getNumJoints(targid)

# for i in range(n):
#     print(p.getJointInfo(targid, i))
#
# for step in range(500):
#     focus_position, _ = p.getBasePositionAndOrientation(targid)
#     p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=focus_position)
#     p.stepSimulation()
#     time.sleep(0.01)

jointid = 4
jtype = p.getJointInfo(targid, jointid)[2]
jlower = p.getJointInfo(targid, jointid)[8]
jupper = p.getJointInfo(targid, jointid)[9]
p.disconnect()

env = ArmEnv()
for step in range(500):
    action1 = np.random.uniform(jlower, jupper)
    action2 = np.random.uniform(jlower, jupper)
    a,b = env.step(action1, action2)
    print(env.state)
    focus_position, _ = p.getBasePositionAndOrientation(targid)
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=focus_position)
    p.stepSimulation()
    time.sleep(0.01)