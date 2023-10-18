import numpy as np
import pybullet as p
import pybullet_data
import time

class ArmEnv():

    def __init__(self):
        p.connect(p.GUI)
        self.state = self.init_state()
        self.step_count = 0

    def init_state(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.loadURDF("/Users/justinlidard/bullet3/examples/pybullet/gym/pybullet_data/plane.urdf", [0, 0, 0], [0, 0, 0, 1])
        self.pandaUid = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        finger_pos = p.getLinkState(self.pandaUid, 9)[0]
        obs = np.array([finger_pos]).flatten()
        return obs

    def reset(self):
        p.resetSimulation()
        self.state = self.init_state()
        self.step_count = 0

    def step(self, action1, action2):
        self.step_count += 1
        p.setJointMotorControlArray(self.pandaUid, [2, 4], p.POSITION_CONTROL, [action1, action2])
        p.stepSimulation()
        finger_pos = p.getLinkState(self.pandaUid, 9)[0]

        if self.step_count >= 50:
            self.reset()
            finger_pos = p.getLinkState(self.pandaUid, 9)[0]
            obs = np.array([finger_pos]).flatten()
            self.state = obs
            reward = -1
            done = True
            return reward, done

        obs = np.array([finger_pos]).flatten()
        self.state = obs
        done = False
        reward = -1
        return reward, done