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
wall1 = p.loadURDF("../object/wall.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
wall2 = p.loadURDF("../object/wall.urdf", [0, 2, 0], [0, 0, 0, 1], useFixedBase=True)
wall3 = p.loadURDF("../object/wall.urdf", [0, 4, 0], [0, 0, 0, 1], useFixedBase=True)
wall4 = p.loadURDF("../object/wall.urdf", [0, 6, 0], [0, 0, 0, 1], useFixedBase=True)

robot = p.loadURDF("sphere2.urdf", [0, -2, 0], [0, 0, 0, 1], useFixedBase=True)



time.sleep(5)