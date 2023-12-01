import gymnasium as gym
from gymnasium import spaces
import numpy as np
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

LEFT_BOUNDARY = -5
RIGHT_BOUNDARY = 5
UPPER_BOUNDARY = 5
LOWER_BOUNDARY = -5

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

WALL_XLEN = 2
WALL_YLEN = 1

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


def collision_with_boundaries(robot_pos):
    if robot_pos[0] <= LEFT_BOUNDARY or robot_pos[0] >= RIGHT_BOUNDARY or \
            robot_pos[1] <= LOWER_BOUNDARY or robot_pos[1] >= UPPER_BOUNDARY:
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
                 urdfRoot=pybullet_data.getDataPath()):
        super(BulletHallwayEnv, self).__init__()

        self.urdfRoot = urdfRoot
        self.cam_dist = 7
        self.cam_yaw = 20
        self.cam_pitch = -45
        self.cam_targ = [0, 0, 0]

        if render:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)

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
        self.action_space = spaces.Box(low=-max_turning_rate, high=max_turning_rate, shape=(2,))
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
        self.prev_corridor_dist = 0

    def state_history_numpy(self):
        state_history = np.array(self.prev_states).flatten()
        return state_history

    def step(self, action):

        self.timesteps += 1

        wall_dist = wall_set_distance(self.walls, self.robot_state)
        human_wall_dist = wall_set_distance(self.walls, self.human_state)

        if self.display_render:
            self.render()

        # controls = np.array([-self.max_turning_rate, 0, self.max_turning_rate])
        # actions = [controls[a] for a in action]
        actions = action
        targetvel = -1
        robot_action = np.array([targetvel, action[0]])
        human_action = np.array([targetvel, action[1]])
        same_action_time = 15
        for _ in range(same_action_time):
            self.robot.applyAction(robot_action)
            self.human.applyAction(human_action)
            self.p.stepSimulation()

        self.human_state = self.get_state(self.human.racecarUniqueId)
        self.robot_state = self.get_state(self.robot.racecarUniqueId)

        truncated = False
        collision_penalty = 0
        wall_coord = self.walls[self.intent]
        wall_left = wall_coord[0]
        wall_right = WALL_XLEN
        wall_up = wall_coord[1] + WALL_YLEN
        wall_down = wall_up - WALL_YLEN
        rect = (wall_left, wall_up, WALL_XLEN, WALL_YLEN)
        wrong_hallway = self.intent_violation(self.human_state, rect)

        violated_dist = any(wall_dist <= 0.25) or any(human_wall_dist <= 0.25)
        if violated_dist:
            self.done = False
            collision_penalty = 0.1

        intent_bonus = 0
        intent_corridor_dist = wall_set_distance([rect[:2]], self.human_state)[0]
        if self.human_state[0] < wall_left:
            intent_bonus = self.prev_corridor_dist - intent_corridor_dist
        self.prev_corridor_dist = intent_corridor_dist
        # print(intent_bonus)

        if collision_with_human(self.robot_state, self.human_state):
            self.done = True
            collision_penalty = 1

        if collision_with_boundaries(self.robot_state) == 1 or collision_with_boundaries(self.human_state) == 1:
            self.done = True
            collision_penalty = 0.1

        if wrong_hallway:
            self.done = True

        if self.timesteps >= self.time_limit:
            self.done = True
            truncated = True

        # Compute reward
        human_intent_mismatch_penalty = 0
        self.dist_robot, _ = distance_to_goal(self.robot_state, self.robot_goal_rect)
        self.dist_human, _ = distance_to_goal(self.human_state, self.human_goal_rect)
        self.robot_distance = np.linalg.norm(self.robot_state[:2] - self.robot_goal_rect[:2])
        self.human_distance = np.linalg.norm(self.human_state[:2] - self.human_goal_rect[:2])
        if self.learning_agent == LearningAgent.HUMAN:
            self.reward = self.prev_dist_human - self.dist_human
            self.reward = np.sign(self.reward)
        else:
            self.reward = self.prev_dist_robot - self.dist_robot
            self.reward = np.sign(self.reward)
        self.reward = self.prev_dist_human - self.dist_human + self.prev_dist_robot - self.dist_robot
        self.reward = self.reward
        #print(self.reward)
        self.reward += - collision_penalty #+ intent_bonus
        self.prev_reward = self.reward
        self.prev_dist_robot = self.dist_robot
        self.prev_dist_human = self.dist_human

        self.cumulative_reward += self.reward


        info = {}

        human_delta_x = self.robot_state[0] - self.human_state[0]
        human_delta_y = self.robot_state[1] - self.human_state[1]

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
                                         human_wall_dist[self.intent:self.intent+1],
                                         robot_wall_dist[self.intent:self.intent+1],
                                         self.robot_state[-1:], self.human_state[-1:],
                                         np.array([self.dist_robot, self.dist_human])))
        #observation = np.concatenate((self.robot_state, self.human_state))
        observation = {"obs": observation, "mode": np.eye(5)[self.intent]}

        #cv2.imwrite('test.png', observation)

        return observation, self.reward, self.done, truncated, info

    def reset(self, seed=1234, options={}):

        self.timesteps = 0
        self.p.resetSimulation()

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
            human_position = np.array([4, 0])
            human_heading = np.pi + np.array(np.pi)
        else:
            human_position = np.array([np.random.uniform(low=3, high=4.5), np.random.uniform(low=-3, high=3)])
            human_heading = np.pi + np.random.uniform(low=3*np.pi/4, high=5*np.pi/4)
        self.human_state = np.array([human_position[0], human_position[1], human_heading], dtype=np.float32)

        if self.debug:
            robot_position = np.array([-4, 0])
            robot_heading = np.pi #+ np.array(np.pi/3)
        else:
            robot_position = np.array([np.random.uniform(low=-4.5, high=-3), np.random.uniform(low=-3, high=3)])
            robot_heading = np.pi + np.random.uniform(low=-np.pi/3, high=np.pi/3)
        self.robot_state = np.array([robot_position[0], robot_position[1], robot_heading], dtype=np.float32)

        human_orientation = self.p.getQuaternionFromEuler([0, 0, human_heading])
        robot_orientation = self.p.getQuaternionFromEuler([0, 0, robot_heading])

        # Load assets
        self.p.setGravity(0, 0, -9.8)

        home = expanduser("~")
        wallpath = os.path.join(home, 'PredictiveRL/object/wall.urdf') #/Users/justinlidard/bullet3/examples/pybullet/gym/pybullet_data/
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = self.p.loadURDF("plane.urdf",
                                     [0, 0, 0],
                                     [0, 0, 0, 1])
        wall1 = self.p.loadURDF(wallpath, [0, -3, 0], [0, 0, 0, 1],
                                useFixedBase=True)
        wall2 = self.p.loadURDF(wallpath, [0, -1, 0], [0, 0, 0, 1],
                                useFixedBase=True)
        wall3 = self.p.loadURDF(wallpath, [0, 1, 0], [0, 0, 0, 1],
                                useFixedBase=True)
        wall4 = self.p.loadURDF(wallpath, [0, 3, 0], [0, 0, 0, 1],
                                useFixedBase=True)


        goal_orientation = self.p.getQuaternionFromEuler([0, 0, -np.pi / 2])
        goal1 = self.p.loadURDF(wallpath, [3, 0, -0.4999], goal_orientation,
                           useFixedBase=True)
        goal2 = self.p.loadURDF(wallpath, [-3, 0, -0.4999], goal_orientation,
                           useFixedBase=True)


        # visualize human intent
        wall_coord = self.walls[self.intent]
        wall_left = wall_coord[0]
        wall_right = WALL_XLEN
        wall_up = wall_coord[1] + WALL_YLEN
        wall_down = wall_up - WALL_YLEN
        rect = (wall_left, wall_up, WALL_XLEN, WALL_YLEN)
        intent = self.p.loadURDF(wallpath, [wall_left + WALL_XLEN/2, wall_up - WALL_YLEN/2, -0.4999], [0, 0, 0, 1],
                                useFixedBase=True)


        self.wall_assets = [wall1, wall2, wall3, wall4]
        self.goal_assets = [goal1, goal2]
        for asset in self.wall_assets:
            self.p.changeVisualShape(asset, -1, rgbaColor=[0, 0, 0, 1])
        self.p.changeVisualShape(goal1, -1, rgbaColor=[0, 0, 1, 1])
        self.p.changeVisualShape(goal2, -1, rgbaColor=[1, 0, 0, 1])
        self.p.changeVisualShape(intent, -1, rgbaColor=[1, 0, 0, 0.5])
        self.human = racecar.Racecar(self.p, urdfRootPath=self.urdfRoot, timeStep=self.timesteps,
                                     pos=(human_position[0],human_position[1],0.2),
                                     orientation=human_orientation)
        self.robot = racecar.Racecar(self.p, urdfRootPath=self.urdfRoot, timeStep=self.timesteps,
                                     pos=(robot_position[0],robot_position[1],0.2),
                                     orientation=robot_orientation)

        self.human.applyAction([0, 0])
        self.robot.applyAction([0, 0])
        for i in range(100):
            self.p.stepSimulation()

        # uplane_orientation = self.p.getQuaternionFromEuler([np.pi/2, 0, 0])
        # upperplane = self.p.loadURDF("/Users/justinlidard/bullet3/examples/pybullet/gym/pybullet_data/plane.urdf",
        #                              [0, 5, 0],
        #                              uplane_orientation)
        # self.p.changeVisualShape(upperplane, -1, rgbaColor=[1, 0, 0, 0.1])
        # rplane_orientation = self.p.getQuaternionFromEuler([0, np.pi/2, 0])
        # rightplane = self.p.loadURDF("/Users/justinlidard/bullet3/examples/pybullet/gym/pybullet_data/plane.urdf",
        #                              [5, 0, 0],
        #                              rplane_orientation)
        # self.p.changeVisualShape(rightplane, -1, rgbaColor=[1, 0, 0, 0.1])
        # lplane_orientation = self.p.getQuaternionFromEuler([0, np.pi/2, 0])
        # leftplane = self.p.loadURDF("/Users/justinlidard/bullet3/examples/pybullet/gym/pybullet_data/plane.urdf",
        #                              [-5, 0, 0],
        #                              lplane_orientation)
        # self.p.changeVisualShape(leftplane, -1, rgbaColor=[1, 0, 0, 0.1])
        # dplane_orientation = self.p.getQuaternionFromEuler([np.pi/2, 0, 0])
        # lowerplane = self.p.loadURDF("/Users/justinlidard/bullet3/examples/pybullet/gym/pybullet_data/plane.urdf",
        #                              [0, -5, 0],
        #                              dplane_orientation)
        # self.p.changeVisualShape(lowerplane, -1, rgbaColor=[1, 0, 0, 0.1])

        self.human_state = self.get_state(self.human.racecarUniqueId)
        self.robot_state = self.get_state(self.robot.racecarUniqueId)

        # Goals
        self.robot_goal_rect = np.array([-3.5, 1, 1, 2])
        self.human_goal_rect = np.array([2.5, 1, 1, 2])
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1

        self.dist_robot, _ = distance_to_goal(self.robot_state, self.robot_goal_rect)
        self.dist_human, _ = distance_to_goal(self.human_state, self.human_goal_rect)
        self.prev_dist_robot = self.dist_robot
        self.prev_dist_human = self.dist_human

        self.done = False

        _, disp = distance_to_goal(self.human_state, self.human_goal_rect)

        self.prev_states= []  # short state history to incorporate some memory in the policy
        for i in range(self.obs_seq_len):
            self.prev_states.append([0] * self.state_dim)  # to create history

        intent_corridor_dist = wall_set_distance([rect[:2]], self.human_state)[0]
        self.prev_corridor_dist = intent_corridor_dist
        self.reward = - np.linalg.norm(self.robot_state[:2] - self.robot_goal_rect[:2]) - np.linalg.norm(self.human_state[:2] - self.human_goal_rect[:2])
        self.prev_reward = self.reward
        self.cumulative_reward = 0

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
                                         human_wall_dist[self.intent:self.intent+1],
                                         robot_wall_dist[self.intent:self.intent+1],
                                         self.robot_state[-1:], self.human_state[-1:],
                                         np.array([self.dist_robot, self.dist_human])))
        #print(observation.shape)
        # observation = np.concatenate((self.robot_state, self.human_state))
        observation = {"obs": observation, "mode": np.eye(5)[self.intent]}

        # from PIL import Image
        # img = Image.fromarray(observation["obs"], 'RGB')
        # img.save('try.png')
        # import time
        # print("something")
        # time.sleep(10)

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
        return rgb_array
        
