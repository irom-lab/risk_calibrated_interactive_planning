import numpy as np
import pybullet as p
import pybullet_data
from arm_env import ArmEnv
import time

import racecar

urdfRoot=pybullet_data.getDataPath()

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)
#time.sleep(5)

p.loadURDF("/Users/justinlidard/bullet3/examples/pybullet/gym/pybullet_data/plane.urdf", [0, 0, 0], [0, 0, 0, 1])
wall1 = p.loadURDF("/Users/justinlidard/PredictiveRL/object/wall.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
wall2 = p.loadURDF("/Users/justinlidard/PredictiveRL/object/wall.urdf", [0, 2, 0], [0, 0, 0, 1], useFixedBase=True)
wall3 = p.loadURDF("/Users/justinlidard/PredictiveRL/object/wall.urdf", [0, 4, 0], [0, 0, 0, 1], useFixedBase=True)
wall4 = p.loadURDF("/Users/justinlidard/PredictiveRL/object/wall.urdf", [0, 6, 0], [0, 0, 0, 1], useFixedBase=True)

goal1 = p.loadURDF("/Users/justinlidard/PredictiveRL/object/wall.urdf", [0, 6, 0], [0, 0, 0, 1], useFixedBase=True)
goal2 = p.loadURDF("/Users/justinlidard/PredictiveRL/object/wall.urdf", [0, 6, 0], [0, 0, 0, 1], useFixedBase=True)

#robot = p.loadURDF("sphere2.urdf", [0, -2, 0], [0, 0, 0, 1], useFixedBase=True)
racecar = racecar.Racecar(p, urdfRootPath=urdfRoot, timeStep=0)



time.sleep(5)