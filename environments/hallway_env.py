import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 30
LEFT_BOUNDARY = 0
RIGHT_BOUNDARY = 1600
UPPER_BOUNDARY = 0
LOWER_BOUNDARY = 900

TIMEOUT_TIMESTEPS = 100
WALL_XLEN = 800
WALL_YLEN = 100


def collision_with_human(robot_position, human_position, eps=1):
    col = np.linalg.norm(robot_position[:2] - human_position[:2]) < eps
    return col

def distance_to_goal(pos, goal_rect):
    return rect_set_dist(goal_rect, pos)


def collision_with_boundaries(robot_pos):
    if robot_pos[0] < LEFT_BOUNDARY or robot_pos[0] >= RIGHT_BOUNDARY or \
            robot_pos[1] <= UPPER_BOUNDARY or robot_pos[1] > LOWER_BOUNDARY:
        return 1
    else:
        return 0

def wall_set_distance(walls, robot_pos):
    wall_dist = []
    for wall_idx in range(len(walls)):
        wall_coord = walls[wall_idx]
        wall_left = wall_coord[0]
        wall_right = wall_left + WALL_XLEN
        wall_up = wall_coord[1]
        wall_down = wall_up + WALL_YLEN

        rect = (wall_left, wall_up, wall_right, wall_down)
        signed_dist, _, _ = rect_set_dist(rect, robot_pos)
        wall_dist.append(signed_dist)
    return np.array(wall_dist)

def rect_set_dist(rect, pos):
    (rect_left, rect_up, rect_right, rect_down) = rect
    dl = rect_left - pos[0]  # left displacement (positive => left of rectangle)
    dr = pos[0] - rect_right  # right distance (positive => right of rectangle)
    du = rect_up - pos[1]  # upper displacement (positive => above rectangle)
    dlow = pos[1] - rect_down  # lower displacement (positive => below rectangle)

    inside_x_coord = dl <= 0 and dr <= 0
    inside_y_coord = du <= 0 and dlow <= 0
    inside = inside_x_coord and inside_y_coord
    dist_sign = -1 if inside else 1
    disp_x = min(abs(dl), abs(dr))
    disp_y = min(abs(du), abs(dlow))
    signed_dist = dist_sign * (disp_x ** 2 + disp_y ** 2) ** (0.5)
    return signed_dist, disp_x, disp_y

def rect_unpack_sides(rect):
    rect_left, rect_up, rect_right, rect_down = rect
    rect_right += rect_left
    rect_down += rect_up
    return (rect_left, rect_up, rect_right, rect_down)

def state_to_tripoints(pos, tripoints_bf):
    heading = pos[-1] - np.pi/2
    rot_mat = np.array([[np.cos(heading), -np.sin(heading)],
                       [np.sin(heading), np.cos(heading)]])
    new_tripoints = rot_mat @ tripoints_bf.T
    new_tripoints = new_tripoints.T
    new_tripoints[:, 1] *= -1
    return new_tripoints + pos[:2]



class HallwayEnv(gym.Env):

    def __init__(self, state_dim=4, obs_seq_len=10):
        super(HallwayEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        self.obs_seq_len = obs_seq_len
        self.state_dim = state_dim
        self.full_obs_dim = self.state_dim + self.obs_seq_len*self.state_dim
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(self.full_obs_dim,), dtype=np.float32)

    def state_history_numpy(self):
        state_history = np.array(self.prev_states).flatten()
        return state_history

    def step(self, action):
        self.timesteps += 1
        cv2.imshow('Hallway Environment', self.img)
        cv2.waitKey(1)
        self.img = 255 - np.zeros((900, 1600, 3), dtype='uint8')

        wall_dist = wall_set_distance(self.walls, self.robot_position)

        robot_tripoints = state_to_tripoints(self.robot_position, self.robot_tripoints)
        # Display human
        cv2.rectangle(self.img, (self.human_position[0], self.human_position[1]),
                      (self.human_position[0] + 10, self.human_position[1] + 10), (0, 0, 255), -1)
        # # Display Robot
        # cv2.rectangle(self.img, (self.robot_position[0], self.robot_position[1]),
        #               (self.robot_position[0] + 10, self.robot_position[1] + 10), (255, 0, 0), -1)
        cv2.fillPoly(self.img, [robot_tripoints.reshape(-1, 1, 2).astype(np.int32)], color=(255, 0, 0))
        # vertices = np.array([[480, 400], [250, 650], [600, 650]], np.int32)
        # pts = vertices.reshape((-1, 1, 2))
        # cv2.polylines(self.img, [pts], isClosed=True, color=(0, 0, 255), thickness=20)

        # Display Goal
        cv2.rectangle(self.img, (self.robot_goal_rect[0], self.robot_goal_rect[1]),
                      (self.robot_goal_rect[0] + self.robot_goal_rect[2],
                       self.robot_goal_rect[1] + self.robot_goal_rect[3]), (255, 0, 0), 3)
        # Display Human Goal
        cv2.rectangle(self.img, (self.human_goal_rect[0], self.human_goal_rect[1]),
                      (self.human_goal_rect[0] + self.human_goal_rect[2],
                       self.human_goal_rect[1] + self.human_goal_rect[3]), (0, 0, 255), 3)

        for wall in self.walls:
            cv2.rectangle(self.img, (wall[0], wall[1]), (wall[0] + WALL_XLEN, wall[1] + WALL_YLEN), (0, 0, 0), -1)

        # Takes step after fixed time
        t_end = time.time() + 0.05
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue

        button_direction = action
        # Change the head position based on the button direction
        # if button_direction == 1:
        #     self.robot_position[0] += 10
        #     self.robot_position[2] = 0
        # elif button_direction == 0:
        #     self.robot_position[0] -= 10
        #     self.robot_position[2] = np.pi
        # elif button_direction == 2:
        #     self.robot_position[1] += 10
        #     self.robot_position[2] = 3/2 * np.pi
        # elif button_direction == 3:
        #     self.robot_position[1] -= 10
        #     self.robot_position[2] = np.pi
        if button_direction == 1:
            self.robot_position[0] += 10*np.cos(self.robot_position[2])
            self.robot_position[1] -= 10*np.sin(self.robot_position[2])
        elif button_direction == 2:
            self.robot_position[2] -= np.pi/10
        elif button_direction == 3:
            self.robot_position[2] += np.pi/10


        # On collision, kill the trial and print the score
        violated_dist = any(wall_dist < 0)
        if collision_with_boundaries(self.robot_position) == 1 or collision_with_human(self.robot_position, self.human_position) \
                or distance_to_goal(self.robot_position, self.robot_goal_rect)[0] < 0 \
                or self.timesteps >= TIMEOUT_TIMESTEPS \
                or violated_dist:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = 255 - np.zeros((900, 1600, 3), dtype='uint8')
            cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow('Hallway Environment', self.img)
            self.done = True

        signed_dist , _, _ = distance_to_goal(self.robot_position, self.robot_goal_rect)
        self.total_reward = -signed_dist
        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward

        if self.done:
            self.reward = 0
        if violated_dist:
            self.reward = np.min(violated_dist)
        info = {}

        _, pos_x, pos_y = distance_to_goal(self.robot_position, self.robot_goal_rect)

        human_delta_x = self.robot_position[0] - self.human_position[0]
        human_delta_y = self.robot_position[1] - self.human_position[1]

        # create observation:

        self.prev_states.pop(0)  # pop the oldest observation
        current_observation = [pos_x, pos_y, human_delta_x, human_delta_y]
        self.prev_states.append(current_observation)
        state_history = self.state_history_numpy()
        observation = np.concatenate((current_observation, state_history), axis=-1)


        return observation, self.reward, self.done, info

    def reset(self):
        self.img = 255 - np.zeros((900, 1600, 3), dtype='uint8')
        self.timesteps = 0
        self.walls = np.array([[400, 100],
                                [400, 300],
                                [400, 500],
                                [400, 700]])
        # Initial robot and human position
        self.robot_position = np.array([1550, 850, np.pi/2], dtype=np.float32)
        self.robot_tripoints = np.array([[0, 25], [-10, -25], [10, -25]])
        self.human_position = np.array([random.randrange(1, 400), random.randrange(1, 900)])
        while(np.min(wall_set_distance(self.walls, self.human_position)) <= 0):
            self.human_position = np.array([random.randrange(1, 400), random.randrange(1, 900)])
        self.robot_goal_rect = np.array([100, 300, 200, 300])
        self.human_goal_rect = np.array([1300, 300, 200, 300])
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1

        self.prev_reward = 0

        self.done = False

        pos_x = self.robot_position[0]
        pos_y = self.robot_position[1]

        human_delta_x = self.human_position[0] - pos_x
        human_delta_y = self.human_position[1] - pos_y

        self.prev_states= []  # short state history to incorporate some memory in the policy
        for i in range(self.obs_seq_len):
            self.prev_states.append([0] * self.state_dim)  # to create history

        state_history = self.state_history_numpy()

        # create observation:
        observation = np.concatenate(([pos_x, pos_y, human_delta_x, human_delta_y], state_history), axis=-1)

        return observation