import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import cv2
import os
from os.path import expanduser
import random
import time
from enum import IntEnum
from collections import deque

import pybullet
from . import racecar
import random
from pybullet_utils import bullet_client as bc
import pybullet_data
from pkg_resources import parse_version

LEFT_BOUNDARY = -7
RIGHT_BOUNDARY = 7
UPPER_BOUNDARY = 4.5
LOWER_BOUNDARY = -4.5

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

WALL_XLEN = 2
WALL_YLEN = 1

# prompt = "This is a simulation of two robots navigating a room with 5 hallways numbered 0-4. " \
#     "The human's robot on the right side of the image. The robot on the left is controlled autonomously." \
#     "Based on what you can see about the human's robot, what can you infer about which hallway it will travel towards?" \
#     "What hallway is the human's robot closest to?" "What hallway is the human's robot pointing towards?" \
#     "Based on the human's current position and heading, which" \
#     "hallway(s) is the human likely to enter? Give approximate numeric probabilities for all hallways 0-4," \
#     " in order starting with 0. Hallway 0 is at the top."

# prompt = "This is a picture of two toy cars navigating a room with 5 hallways numbered 0-4. " \
#     "The cars always go forward and can turn left and right. They want to get to the other side." \
#     "Based on the right car's current position and heading, which" \
#     "hallways are the right car likely to enter? Give approximate numeric probabilities for all hallways 0-4," \
#     " in order starting with 0. Hallway 0 is at the top."

prompt = "This is a metaphorical scenario of two toy cars navigating a room with 5 hallways numbered 0-4. " \
    "This is a set of images from the scenario." \
    "The Ego car is current heading towards a set of hallways, and could take any of them." \
    "Based on the Ego car's trajectory, which of the hallways is it going towards?" \
    "Give approximate numerical probabilities for all hallways 0-4 as a bulleted list and explain each one." \
    "Never give higher than 80% and lower than 10% since it's always best to be a little skeptical." \
    "Always give adjacent hallways to your most-likely prediction a near-equal weight"

# prompt = "This is a metaphorical scenario of two toy cars navigating a room with 5 hallways numbered 0-4. " \
#     "The walls of the hallways are depicted in black." \
#     "Hallway 0 is at the top of the screen. Hallway 1 is directly underneath hallway 0, in the top middle." \
#     "Hallway 2 is directly underneath hallway 1, in the middle. Hallway 3 is directly underneath hallway 2" \
#     "in the bottom middle. Hallway 4 is direct underneath hallway 3 at the bottom of the image." \
#     "This is a set of images from the scenario. Let's call the car on the right half of the screen the Ego Car." \
#     "The Ego Car is currently heading towards a set of hallways, and could take any of them." \
#     "You can use the Ego Car's position over time as evidence for estimating which hallway it's going to." \
#     "Describe the Ego Car's position over time, which number is it heading towards?" \
#     "Now, give approximate numerical probabilities for all numbers 0-4 as a bulleted list and explain each one."

# prompt = "This is a metaphorical scenario of two toy cars navigating a room with 5 paths numbered 0-4. " \
#     "This is a set of images from the scenario. Let's call the car on the right half of the screen the Ego Car." \
#     "The Ego Car is currently heading towards a set of paths, and could take any of them." \
#     "Give approximate numerical probabilities for each one."






class HumanIntent(IntEnum):
    HALLWAY1 = 0
    HALLWAY2 = 1
    HALLWAY3 = 2
    HALLWAY4 = 3
    HALLWAY5 = 4

class LearningAgent(IntEnum):
    HUMAN=0 # Red
    ROBOT=1 # Blue

def collision_with_human(robot_position, human_position, eps=1):
    col = np.linalg.norm(robot_position[:2] - human_position[:2]) < eps
    return col

def distance_to_goal(pos, goal_rect):
    return rect_set_dist(goal_rect, pos)

def distance_to_goal_l2(pos, goal):
    return np.linalg.norm(pos[:2] - goal[:2])


def collision_with_boundaries(robot_pos, eps=0.76):
    if robot_pos[0] <= LEFT_BOUNDARY+eps or robot_pos[0] >= RIGHT_BOUNDARY-eps: #robot_pos[1] <= LOWER_BOUNDARY+eps or robot_pos[1] >= UPPER_BOUNDARY-eps or \
        return 1
    else:
        return 0

def wall_set_distance(walls, robot_pos):
    wall_dist = []
    for wall_idx in range(len(walls)):
        wall_coord = walls[wall_idx]
        wall_left = wall_coord[0]
        wall_right = WALL_XLEN
        wall_up = wall_coord[1]
        wall_down = WALL_YLEN

        rect = (wall_left, wall_up, wall_right, wall_down)
        signed_dist, _ = rect_set_dist(rect, robot_pos)
        wall_dist.append(signed_dist)
    return np.array(wall_dist)

def hallway_l2_dist(rect, pos):
    left, up, dx, dy = rect
    right = left + dx
    down = up - dy
    centroid = np.array([(left+right)/2, (up+down)/2])
    dist = np.linalg.norm(pos[:2]-centroid)
    return dist

def rect_set_dist(rect, pos):
    (rect_left, rect_up, drect_right, drect_down) = rect
    rect_down = rect_up - drect_down
    rect_right = rect_left + drect_right

    # compute displacements (positive ==> on "this" side, e.g. dr>0 ==> on the right side, du>0 ==> above, etc.)
    # if both components of an axis are negative (e.g. up and down), then pos is inside the rectangle on this axis
    dl = rect_left - pos[0]  # left displacement (positive => left of rectangle)
    dr = pos[0] - rect_right  # right distance (positive => right of rectangle)
    du = pos[1] - rect_up # upper displacement (positive => above rectangle)
    dlow = rect_down - pos[1]  # lower displacement (positive => below rectangle)

    inside_x_coord = dl <= 0 and dr <= 0
    inside_y_coord = du <= 0 and dlow <= 0
    inside = inside_x_coord and inside_y_coord
    dist_sign = -1 if inside else 1
    disp_x = min(abs(dl), abs(dr))
    disp_y = min(abs(du), abs(dlow))

    if inside_x_coord and not inside_y_coord:
        signed_dist = dist_sign * disp_y
    elif inside_y_coord and not inside_x_coord:
        signed_dist = dist_sign * disp_x
    elif inside_x_coord and inside_y_coord:
        signed_dist = 0
    else:
        signed_dist = dist_sign * (disp_x ** 2 + disp_y ** 2) ** (0.5)
    return signed_dist, (dl, dr, du, dlow)

def rect_unpack_sides(rect):
    rect_left, rect_up, rect_right, rect_down = rect
    rect_right += rect_left
    rect_down += rect_up
    return (rect_left, rect_up, rect_right, rect_down)


class BulletHallwayEnv(gym.Env):

    def __init__(self, render=False, state_dim=6, obs_seq_len=10, max_turning_rate=1, deterministic_intent=None,
                 debug=False, render_mode="rgb_array", time_limit=100, rgb_observation=False,
                 urdfRoot=pybullet_data.getDataPath(), show_intent=True,history_log_path=None):
        super(BulletHallwayEnv, self).__init__()
        self.show_intent = show_intent
        self.log_history = (history_log_path is not None)
        self.df_savepath = history_log_path
        self.urdfRoot = urdfRoot
        self.cam_dist = 8
        self.cam_yaw = 60 #-60
        self.cam_pitch = -45
        self.cam_targ = [0, 0, 0]

        if render:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.time_step_dt = 0.01
        pybullet.setTimeStep(self.time_step_dt)

        self.p.resetDebugVisualizerCamera(cameraDistance=self.cam_dist,
                                          cameraYaw=self.cam_yaw,
                                          cameraPitch=self.cam_pitch,
                                          cameraTargetPosition=self.cam_targ)
        self.p.resetSimulation()

        self.robot = None
        self.human = None

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        num_latent_vars = len(HumanIntent)
        self.num_latent_vars = num_latent_vars
        self.max_turning_rate = max_turning_rate
        self.action_space = spaces.Box(low=-max_turning_rate, high=max_turning_rate, shape=(4,))
        # self.action_space = spaces.MultiDiscrete(nvec=[3, 3])
        self.obs_seq_len = obs_seq_len
        self.state_dim = state_dim
        self.full_obs_dim = self.state_dim + self.obs_seq_len*self.state_dim
        self.rgb_observation = rgb_observation
        if self.rgb_observation:
            self.observation_space = {"obs": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                                      "mode": spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)}
        else:
            self.observation_space = {"obs": spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32),
                                      "mode": spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)}
                                  # "agent": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)}
        self.observation_space = spaces.Dict(self.observation_space)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.intent = None
        self.deterministic_intent = deterministic_intent

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.display_render = render
        self.done = False
        self.debug = debug
        self.render_mode = render_mode
        self.intent_seed = None
        self.dist_robot = self.dist_human = 0
        self.prev_dist_robot = self.prev_dist_human = 0
        self.time_limit = time_limit
        self.learning_agent = LearningAgent.HUMAN
        self.cumulative_reward = 0
        self.stuck_counter = 0
        self.prev_human_hallway_dist = 0
        self.prev_robot_hallway_dist = 0

        # Load assets
        self.p.setGravity(0, 0, -9.8)

        home = expanduser("~")
        wallpath = os.path.join(home, 'PredictiveRL/object/wall.urdf') #/Users/justinlidard/bullet3/examples/pybullet/gym/pybullet_data/
        self.wallpath = wallpath
        boundarylongpath = os.path.join(home, 'PredictiveRL/object/boundary_long.urdf')
        boundaryshortpath = os.path.join(home, 'PredictiveRL/object/boundary_short.urdf')
        goalpath = os.path.join(home, 'PredictiveRL/object/goal.urdf')
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = self.p.loadURDF("plane.urdf",
                                     [0, 0, 0],
                                     [0, 0, 0, 1])
        wall1 = self.p.loadURDF(wallpath, [0, -3, 0], [0, 0, 0, 1],
                                useFixedBase=True, useMaximalCoordinates=False)
        wall2 = self.p.loadURDF(wallpath, [0, -1, 0], [0, 0, 0, 1],
                                useFixedBase=True, useMaximalCoordinates=False)
        wall3 = self.p.loadURDF(wallpath, [0, 1, 0], [0, 0, 0, 1],
                                useFixedBase=True, useMaximalCoordinates=False)
        wall4 = self.p.loadURDF(wallpath, [0, 3, 0], [0, 0, 0, 1],
                                useFixedBase=True, useMaximalCoordinates=False)

        # Goals
        self.human_goal_rect= np.array([-5, 2, 2, 4])
        self.robot_goal_rect = np.array([3, 2, 2, 4])
        goal_orientation = self.p.getQuaternionFromEuler([0, 0, -np.pi / 2])
        goal1 = self.p.loadURDF(goalpath, [self.robot_goal_rect[0]+1, 0, -0.4999], goal_orientation,
                           useFixedBase=True)
        goal2 = self.p.loadURDF(goalpath, [self.human_goal_rect[0]+1, 0, -0.4999], goal_orientation,
                           useFixedBase=True)
        self.wall_assets = [wall1, wall2, wall3, wall4]
        self.goal_assets = [goal1, goal2]
        self.intent_asset = None
        for asset in self.wall_assets:
            self.p.changeVisualShape(asset, -1, rgbaColor=[0, 0, 0, 1])
        if self.show_intent:
            goal_alpha = 1
        else:
            goal_alpha = 0
        self.p.changeVisualShape(goal1, -1, rgbaColor=[0, 0, 1, goal_alpha])
        self.p.changeVisualShape(goal2, -1, rgbaColor=[1, 0, 0, goal_alpha])


        # Boundaries
        uplane_orientation = lplane_orientation = self.p.getQuaternionFromEuler([np.pi/2, 0, 0])
        ub = self.p.loadURDF(boundarylongpath,[0, UPPER_BOUNDARY+WALL_YLEN/2, 0], uplane_orientation, useFixedBase=True)
        lb = self.p.loadURDF(boundarylongpath,[0, LOWER_BOUNDARY-WALL_YLEN/2, 0], lplane_orientation, useFixedBase=True)

        rplane_orientation = leftplane_orientation = self.p.getQuaternionFromEuler([np.pi/2, 0, 0])
        leftb = self.p.loadURDF(boundaryshortpath,[LEFT_BOUNDARY, 0, 0], rplane_orientation, useFixedBase=True)
        rightb = self.p.loadURDF(boundaryshortpath,[RIGHT_BOUNDARY,0, 0], leftplane_orientation, useFixedBase=True)
        self.boundary_assets = [ub, lb, leftb, rightb]
        self.tid = None
        self.rollout_counter = 0
        self.reset_state_history()

    def step(self, action):

        self.timesteps += 1

        if self.display_render:
            self.render()

        self.compute_state_transition(action)

        # Compute a bonus/penalty for the agents going to the optimal/suboptimal hallway
        intent_bonus = 0
        human_hallway_dist, human_hallway, wrong_hallway = self.compute_dist_to_hallways(is_human=True)
        robot_best_hallway_dist, robot_hallway, i_best = self.compute_dist_to_hallways(is_human=False)
        if self.human_state[0] > self.walls[0][0]:
            intent_bonus += (self.prev_human_hallway_dist - human_hallway_dist).item()
        if self.robot_state[0] < self.walls[0][0] + WALL_XLEN:
            intent_bonus += (self.prev_robot_hallway_dist - robot_best_hallway_dist).item()
        self.prev_robot_hallway_dist = robot_best_hallway_dist
        self.prev_human_hallway_dist = human_hallway_dist

        # Compute reward
        self.compute_reward(wrong_hallway, intent_bonus)
        self.prev_dist_robot = self.dist_robot
        self.prev_dist_human = self.dist_human

        info = {}
        truncated = False
        observation = self.compute_observation(human_hallway)

        if self.log_history:
            self.append_state_history()
            if self.done:
                self.save_state_history()

        if self.display_render:
            if self.tid is not None:
                self.p.removeUserDebugItem(self.tid)
            self.tid = self.p.addUserDebugText(f"Reward: {self.cumulative_reward}", [0, 0, 2], textSize=5, textColorRGB=[0, 0, 0])

        return observation, self.reward, self.done, truncated, info

    def compute_state_transition(self, action):
        action_scale = 1
        action_offset = -2
        agent_0_vel = action_offset  #+action[2]
        agent_1_vel = action_offset  #+action[3]
        robot_action = np.array([agent_0_vel, action_scale*action[0]])
        human_action = np.array([agent_1_vel, action_scale*action[1]])
        same_action_time = 5
        for _ in range(same_action_time):
            self.robot.applyAction(robot_action)
            self.human.applyAction(human_action)
            self.p.stepSimulation()

        self.human_state = self.get_state(self.human.racecarUniqueId)
        self.robot_state = self.get_state(self.robot.racecarUniqueId)

    def compute_dist_to_hallways(self, is_human=True):

        if is_human:
            wall_coord = self.walls[self.intent]
            wall_left = wall_coord[0]
            wall_up = wall_coord[1] + WALL_YLEN
            rect = (wall_left, wall_up, WALL_XLEN, WALL_YLEN)
            wrong_hallway = self.intent_violation(self.human_state, rect)
            human_intent_hallway_dist = hallway_l2_dist(rect, self.human_state) # wall_set_distance([rect[:2]], self.human_state)[0]
            return human_intent_hallway_dist, rect, wrong_hallway
        else:
            robot_closest_wall_dist = np.inf
            i_best = -1
            best_rect = None
            for i, other_wall in enumerate(self.walls):
                if i == self.intent:
                    continue
                wall_left = other_wall[0]
                wall_right = WALL_XLEN
                wall_up = other_wall[1] + WALL_YLEN
                other_rect = (wall_left, wall_up, WALL_XLEN, WALL_YLEN)
                robot_intent_hallway_dist = hallway_l2_dist(other_rect, self.robot_state) # wall_set_distance(([other_rect[:2]]), self.robot_state)
                if robot_intent_hallway_dist < robot_closest_wall_dist:
                    robot_closest_wall_dist = robot_intent_hallway_dist
                    i_best = i
                    best_rect = other_rect
            return robot_closest_wall_dist, best_rect, i_best
        
    def compute_reward(self, wrong_hallway, intent_bonus):

        wall_dist = wall_set_distance(self.walls, self.robot_state)
        human_wall_dist = wall_set_distance(self.walls, self.human_state)

        violated_dist = any(wall_dist <= 0.25) or any(human_wall_dist <= 0.25)
        collision_penalty = 0
        if violated_dist:
            self.done = False
            collision_penalty += 0.001

        if collision_with_human(self.robot_state, self.human_state):
            self.done = True

        if collision_with_boundaries(self.robot_state) == 1 or collision_with_boundaries(self.human_state) == 1:
            self.done = False
            collision_penalty += 0.000

        if wrong_hallway:
            self.done = True

        if self.timesteps >= self.time_limit:
            self.done = True
            truncated = True
        
        self.dist_robot, _ = distance_to_goal(self.robot_state, self.robot_goal_rect)
        self.dist_human, _ = distance_to_goal(self.human_state, self.human_goal_rect)
        human_reach_bonus = 0.1 if self.dist_human == 0 else 0
        robot_reach_bonus = 0.1 if self.dist_robot == 0 else 0
        reach_bonus = human_reach_bonus + robot_reach_bonus
        self.robot_distance = np.linalg.norm(self.robot_state[:2] - self.robot_goal_rect[:2])
        self.human_distance = np.linalg.norm(self.human_state[:2] - self.human_goal_rect[:2])
        self.reward = self.prev_dist_human - self.dist_human + self.prev_dist_robot - self.dist_robot
        self.reward += - collision_penalty + reach_bonus + intent_bonus \

        # Update running totals
        self.prev_reward = self.reward
        self.cumulative_reward += self.reward

    def compute_observation(self, hallway):

        if self.rgb_observation:
            self.get_image(resolution_scale=1)
            observation = cv2.resize(self.img, (256, 256))
        else:
            human_delta = self.robot_state - self.human_state
            human_delta_pos = np.linalg.norm(human_delta[:2])
            human_delta_bearing = np.arctan2(human_delta[0], human_delta[1])
            human_wall_dist = wall_set_distance(self.walls, self.human_state)
            robot_wall_dist = wall_set_distance(self.walls, self.robot_state)
            self.dist_robot, _ = distance_to_goal(self.robot_state, self.robot_goal_rect)
            self.dist_human, _ = distance_to_goal(self.human_state, self.human_goal_rect)
            observation = np.concatenate((np.array([human_delta_pos, human_delta_bearing]),
                                         human_wall_dist, robot_wall_dist,
                                         np.array([hallway_l2_dist(hallway, self.human_state),
                                                  hallway_l2_dist(hallway, self.robot_state)]),
                                         self.robot_state[-1:], self.human_state[-1:],
                                         np.array([self.dist_robot, self.dist_human])))
        observation = {"obs": observation, "mode": np.eye(5)[self.intent]}
        return observation


    def reset(self, seed=1234, options={}):

        self.timesteps = 0
        # if self.human is not None and self.robot is not None:
        #     self.p.removeBody(self.human.racecarUniqueId)
        #     self.p.removeBody(self.robot.racecarUniqueId)
        #     self.p.removeBody(self.intent_asset)

        learning_agent = np.random.choice(2)
        if learning_agent == LearningAgent.ROBOT:
            self.learning_agent = LearningAgent.ROBOT
        else:
            self.learning_agent = LearningAgent.HUMAN

        if self.deterministic_intent is None:
            intent = np.random.choice(self.num_latent_vars)
        else:
            intent = self.deterministic_intent

        if self.intent_seed:
            intent = self.intent_seed

        # if self.debug:
        self.intent = HumanIntent(intent)


        self.img = 255 - np.zeros((900, 1600, 3), dtype='uint8')

        # Hallway boundaries. Top left corner.
        self.walls = np.array([[-1, -4.5],
                               [-1, -2.5],
                               [-1, -0.5],
                               [-1, 1.5],
                               [-1, 3.5]]) # include a "hidden" wall for visualizing the intent
        self.walls = self.walls[::-1]

        # Initial robot and human position
        if self.debug:
            human_position = np.array([6.5, 0])
            human_heading = np.pi + np.pi #np.random.uniform(low=3*np.pi/4, high=5*np.pi/4)
        else:
            human_position = np.array([np.random.uniform(low=5.5, high=6.5), np.random.uniform(low=-2, high=2)])
            human_heading = 0  #+ np.random.uniform(low=3*np.pi/4, high=5*np.pi/4)
        self.human_state = np.array([human_position[0], human_position[1], human_heading], dtype=np.float32)

        if self.debug:
            robot_position = np.array([-6.5, 0])
            robot_heading = np.pi # + np.random.uniform(low=-np.pi/4, high=np.pi/4)
        else:
            robot_position = np.array([np.random.uniform(low=-6.5, high=-5.5), np.random.uniform(low=-2, high=2)])
            robot_heading = np.pi #+ np.random.uniform(low=-np.pi/4, high=np.pi/4)
        self.robot_state = np.array([robot_position[0], robot_position[1], robot_heading], dtype=np.float32)

        human_orientation = self.p.getQuaternionFromEuler([0, 0, human_heading])
        robot_orientation = self.p.getQuaternionFromEuler([0, 0, robot_heading])

        # visualize human intent
        wall_coord = self.walls[self.intent]
        wall_left = wall_coord[0]
        wall_right = WALL_XLEN
        wall_up = wall_coord[1] + WALL_YLEN
        wall_down = wall_up - WALL_YLEN
        rect = (wall_left, wall_up, WALL_XLEN, WALL_YLEN)

        if self.human is None:
            self.human = racecar.Racecar(self.p, urdfRootPath=self.urdfRoot, timeStep=self.timesteps,
                                         pos=(human_position[0],human_position[1],0.2),
                                         orientation=human_orientation)
            self.robot = racecar.Racecar(self.p, urdfRootPath=self.urdfRoot, timeStep=self.timesteps,
                                         pos=(robot_position[0],robot_position[1],0.2),
                                         orientation=robot_orientation)
            if self.show_intent:
                intent = self.p.loadURDF(self.wallpath, [wall_left + WALL_XLEN / 2, wall_up - WALL_YLEN / 2, -0.4999],
                                         [0, 0, 0, 1],
                                         useFixedBase=True)
                self.p.changeVisualShape(intent, -1, rgbaColor=[1, 0, 0, 0.5])
                self.intent_asset = intent

            collisionFilterGroup = 0
            collisionFilterMask = 0
            self.p.setCollisionFilterGroupMask(self.human.racecarUniqueId, -1, collisionFilterGroup, collisionFilterMask)
            self.p.setCollisionFilterGroupMask(self.robot.racecarUniqueId, -1, collisionFilterGroup, collisionFilterMask)

            enableCollision = 1
            for wall in self.wall_assets:
                self.p.setCollisionFilterPair(wall, self.human.racecarUniqueId, -1, -1, enableCollision)
                self.p.setCollisionFilterPair(wall, self.robot.racecarUniqueId, -1, -1, enableCollision)

            for boundary in self.boundary_assets:
                self.p.setCollisionFilterPair(boundary, self.human.racecarUniqueId, -1, -1, enableCollision)
                self.p.setCollisionFilterPair(boundary, self.robot.racecarUniqueId, -1, -1, enableCollision)
        else:
            self.p.resetBasePositionAndOrientation(self.human.racecarUniqueId,(human_position[0],human_position[1],0.2),human_orientation)
            self.p.resetBasePositionAndOrientation(self.robot.racecarUniqueId,(robot_position[0],robot_position[1],0.2),robot_orientation)
            if self.show_intent:
                self.p.resetBasePositionAndOrientation(self.intent_asset, [wall_left + WALL_XLEN / 2, wall_up - WALL_YLEN / 2, -0.4999], [0, 0, 0, 1])


        self.human.applyAction([0, 0])
        self.robot.applyAction([0, 0])
        for i in range(100):
            self.p.stepSimulation()

        self.human_state = self.get_state(self.human.racecarUniqueId)
        self.robot_state = self.get_state(self.robot.racecarUniqueId)


        self.reward = 0
        self.cumulative_reward = 0

        self.dist_robot, _ = distance_to_goal(self.robot_state, self.robot_goal_rect)
        self.dist_human, _ = distance_to_goal(self.human_state, self.human_goal_rect)
        self.prev_dist_robot = self.dist_robot
        self.prev_dist_human = self.dist_human

        self.done = False

        human_hallway_dist, human_hallway, wrong_hallway = self.compute_dist_to_hallways(is_human=True)
        robot_best_hallway_dist, robot_hallway, i_best = self.compute_dist_to_hallways(is_human=False)
        self.prev_robot_hallway_dist = robot_best_hallway_dist
        self.prev_human_hallway_dist = human_hallway_dist

        observation = self.compute_observation(human_hallway)

        if self.log_history:
            self.append_state_history()
            if self.done:
                self.save_state_history()

        return observation, {}

    def get_state(self, agentid):
        pos_orientation = self.p.getBasePositionAndOrientation(agentid)
        p, o = pos_orientation
        o_eul = self.p.getEulerFromQuaternion(o)
        state = [p[0], p[1], o_eul[-1]]
        return np.array(state)

    def seed_intent(self, intent):
        self.intent = intent
        self.intent_seed = intent

        # TODO(justin.lidard): encapsulate this subroutine
        wall_coord = self.walls[self.intent]
        wall_left = wall_coord[0]
        wall_right = WALL_XLEN
        wall_up = wall_coord[1] + WALL_YLEN
        wall_down = wall_up - WALL_YLEN
        rect = (wall_left, wall_up, WALL_XLEN, WALL_YLEN)

        if self.show_intent:
            self.p.resetBasePositionAndOrientation(self.intent_asset,
                                                   [wall_left + WALL_XLEN / 2, wall_up - WALL_YLEN / 2, -0.4999],
                                                   [0, 0, 0, 1])

    def dynamics(self, state, other_state, control, is_human=False):
        return self.dubins_car(state, other_state, control, is_human=is_human)

    def dubins_car(self, state, other_state, control, is_human=False, dt=0.1, V=100, turning_rate_scale=1):

        # Unitless -> rad/s
        turning_rate_modifier = 2 * np.pi * turning_rate_scale
        
        x = state[0]
        y = state[1]
        theta = state[2]

        dx = V * np.cos(theta)
        dy = V * np.sin(theta)
        dtheta = control
        
        xnew = x + dx*dt
        ynew = y - dy*dt

        # Don't allow collisions, but allow the robot to turn around.
        wall_coord = self.walls[self.intent]
        wall_left = wall_coord[0]
        wall_right = WALL_XLEN
        wall_up = wall_coord[1] - WALL_YLEN
        wall_down = wall_up + WALL_YLEN
        rect = (wall_left, wall_up, WALL_XLEN, WALL_YLEN)

        new_state = np.array([xnew, ynew])
        wall_dist = wall_set_distance(self.walls, new_state)
        violated_dist = any(wall_dist <= 0)
        valid_transition = True
        if collision_with_boundaries(new_state) or violated_dist or collision_with_human(new_state, other_state): #or intent_violation(new_state):
            new_state_x = np.array([xnew, y])
            new_state_y = np.array([x, ynew])
            wall_dist_x = wall_set_distance(self.walls, new_state_x)
            violated_dist_x = any(wall_dist_x <= 0)
            wall_dist_y = wall_set_distance(self.walls, new_state_y)
            violated_dist_y = any(wall_dist_y <= 0)
            if not(collision_with_boundaries(new_state_x) == 1 or violated_dist_x or collision_with_human(new_state_x, other_state)):# or intent_violation(new_state_x)):
                ynew = y
            elif not(collision_with_boundaries(new_state_y) == 1 or violated_dist_y or collision_with_human(new_state_y, other_state)):# or intent_violation(new_state_y)):
                xnew = x
            else:
                xnew = x
                ynew = y
            valid_transition = False


        thetanew = (theta + dtheta * dt)
        if thetanew > 2*np.pi:
            thetanew = thetanew - 2*np.pi
        elif thetanew < 0:
            thetanew = thetanew + 2*np.pi

        
        return np.array([xnew, ynew, thetanew])

    def intent_violation(self, pstate, rect, is_human=True):
        if not is_human:
            return False

        signed_dist, disp = rect_set_dist(rect, pstate)
        (dl, dr, du, dlow) = disp
        if dl < 0 and dr < 0:
            human_intent_mismatch = max(dlow, 0) + max(du, 0)
            if is_human and human_intent_mismatch > 0:
                return True
            else:
                return False

        return False


    def render(self):
        # if mode != "rgb_array":
        #     return np.array([])
        base_pos, orn = self.p.getBasePositionAndOrientation(self.robot.racecarUniqueId)
        view_matrix = self.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.cam_targ,
                                                                distance=self.cam_dist,
                                                                yaw=self.cam_yaw,
                                                                pitch=self.cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self.p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (w, h, px, _, _) = self.p.getCameraImage(width=RENDER_WIDTH,
                                                  height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = np.reshape(rgb_array, (h, w, 4))
        #rgb_array = np.transpose(rgb_array, (2, 0, 1))
        rgb_array = rgb_array[:, :, :3]
        img = rgb_array.astype(np.uint8).copy()

        strx = 475
        yoffset = 112
        ydelta = 130
        for i in range(5):
            stry = yoffset + ydelta*i
            str = f"{i}"
            cv2.putText(img, str, (strx, stry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (1, 0, 0), 2, cv2.LINE_AA)

        strx = -int(self.robot_state[0]*70) + 475
        stry = 330 #-int(self.robot_state[1]*70) + 270
        # img = img.astype(np.uint8).copy()
        # str = f"Ego Car"
        # cv2.putText(img, str, (strx, stry),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

        return img #[:, img.shape[0]//2:]


    def save_state_history(self):

        all_robot_state = np.stack(self.robot_state_history)
        all_human_state = np.stack(self.human_state_history)
        all_intent = np.stack(self.intent_history)
        df = pd.DataFrame({"robot_state_x": all_robot_state[:, 0],
                           "robot_state_y": all_robot_state[:, 1],
                           "robot_state_heading": all_robot_state[:, 2],
                           "human_state_x": all_human_state[:, 0],
                           "human_state_y": all_human_state[:, 1],
                           "human_state_heading": all_human_state[:, 2],
                           "human_intent": all_intent})
        os.makedirs(f"{self.df_savepath}/rollouts/", exist_ok=True)
        df.to_csv(f"{self.df_savepath}/rollouts/rollout_{self.rollout_counter}.csv")
        self.reset_state_history()

    def append_state_history(self):
        self.robot_state_history.append(self.robot_state)
        self.human_state_history.append(self.human_state)
        self.intent_history.append(self.intent)

    def reset_state_history(self):
        self.robot_state_history = []
        self.human_state_history = []
        self.intent_history = []
        self.rollout_counter += 1



