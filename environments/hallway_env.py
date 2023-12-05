import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from enum import IntEnum
from collections import deque

SNAKE_LEN_GOAL = 30
LEFT_BOUNDARY = 0
RIGHT_BOUNDARY = 1600
UPPER_BOUNDARY = 0
LOWER_BOUNDARY = 900

WALL_XLEN = 400
WALL_YLEN = 100

# VLM prompt for this env
prompt = "This is a metaphorical and fictitious cartoon of a human navigating a room with 5 hallways numbered 0-4. " \
          "The human is represented by the red triangle. The human's heading is given by the pointy end of the "\
          "traingle. The human could enter one of five numbered hallways in the center of the screen. The human will"\
        "prefer hallways that are closer and aligned with the human's heading. What hallway is the human closest to?" \
         "What hallway is the human pointing towards? Based on the human's current position and heading, which" \
         "hallway(s) is the human likely to enter? Give approximate numeric probabilities for all hallways 0-4," \
          " in order starting with 0. "

# Just give the probabilities and be as brief as possible."

class HumanIntent(IntEnum):
    HALLWAY1 = 0
    HALLWAY2 = 1
    HALLWAY3 = 2
    HALLWAY4 = 3
    HALLWAY5 = 4

class LearningAgent(IntEnum):
    HUMAN=0 # Red
    ROBOT=1 # Blue

def collision_with_human(robot_position, human_position, eps=100):
    col = np.linalg.norm(robot_position[:2] - human_position[:2]) < eps
    return col

def distance_to_goal(pos, goal_rect):
    return rect_set_dist(goal_rect, pos)


def collision_with_boundaries(robot_pos):
    if robot_pos[1] >= LOWER_BOUNDARY or robot_pos[1] <= UPPER_BOUNDARY or \
            robot_pos[0] <= LEFT_BOUNDARY or robot_pos[0] >= RIGHT_BOUNDARY:
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
    rect_down = rect_up + drect_down
    rect_right = rect_left + drect_right

    # compute displacements (positive ==> on "this" side, e.g. dr>0 ==> on the right side, du>0 ==> above, etc.)
    # if both components of an axis are negative (e.g. up and down), then pos is inside the rectangle on this axis
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

def state_to_tripoints(pos, tripoints_bf):
    heading = pos[-1] - np.pi/2
    rot_mat = np.array([[np.cos(heading), -np.sin(heading)],
                       [np.sin(heading), np.cos(heading)]])
    new_tripoints = rot_mat @ tripoints_bf.T
    new_tripoints = new_tripoints.T
    new_tripoints[:, 1] *= -1
    return new_tripoints + pos[:2]



class HallwayEnv(gym.Env):

    def __init__(self, render=False, state_dim=6, obs_seq_len=10, max_turning_rate=5, deterministic_intent=None,
                 debug=False, render_mode="rgb_array", time_limit=100, rgb_observation=False):
        super(HallwayEnv, self).__init__()
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
        self.robot_state = self.dynamics(state=self.robot_state, other_state=self.human_state, control=actions[0])
        self.human_state = self.dynamics(state=self.human_state, other_state=self.robot_state, control=actions[1], is_human=True)

        truncated = False
        collision_penalty = 0
        wall_coord = self.walls[self.intent]
        wall_left = wall_coord[0]
        wall_right = WALL_XLEN
        wall_up = wall_coord[1] - WALL_YLEN
        wall_down = wall_up + WALL_YLEN
        rect = (wall_left, wall_up, WALL_XLEN, WALL_YLEN)
        wrong_hallway = self.intent_violation(self.human_state, rect)

        violated_dist = any(wall_dist <= 10) or any(human_wall_dist <= 10)
        if violated_dist:
            self.done = False
            collision_penalty += 0.05

        if collision_with_human(self.robot_state, self.human_state):
            self.done = True
            collision_penalty += 1

        if collision_with_boundaries(self.robot_state) == 1 or collision_with_boundaries(self.human_state) == 1:
            self.done = False
            collision_penalty += 0.05

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
        self.reward = self.reward / 100
        #print(self.reward)
        self.reward += - collision_penalty
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
            observation = np.concatenate((np.array([human_delta_pos, human_delta_bearing*100]),
                                         human_wall_dist, robot_wall_dist,
                                         wall_set_distance([rect], self.human_state),
                                         wall_set_distance([rect], self.robot_state),
                                         self.robot_state[-1:]*100, self.human_state[-1:]*100,
                                         np.array([self.dist_robot, self.dist_human])))
            observation /= 100
        #observation = np.concatenate((self.robot_state, self.human_state))
        observation = {"obs": observation, "mode": np.eye(5)[self.intent]}

        #cv2.imwrite('test.png', observation)

        wall_set_distance([rect], self.human_state),
        wall_set_distance([rect], self.robot_state),
        return observation, self.reward, self.done, truncated, info

    def reset(self, seed=1234, options={}):

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
        # intent = 1
        self.intent = HumanIntent(intent)


        self.img = 255 - np.zeros((900, 1600, 3), dtype='uint8')
        self.timesteps = 0

        # Hallway boundaries. Top left corner.
        self.walls = np.array([[600, 100],
                                [600, 300],
                                [600, 500],
                                [600, 700],
                                [600, 900]]) # include a "hidden" wall for visualizing the intent

        # Initial robot and human position
        if self.debug:
            robot_position = np.array([1450, 450])
            robot_heading = np.array(np.pi)
        else:
            robot_position = np.array([random.randrange(1300, 1500), random.randrange(300, 600)])
            robot_heading = np.random.uniform(low=3*np.pi/4, high=5*np.pi/4)
        self.robot_state = np.array([robot_position[0], robot_position[1], robot_heading], dtype=np.float32)
        self.robot_tripoints = np.array([[0, 25], [-10, -25], [10, -25]])

        if self.debug:
            human_position = np.array([200, 450])
            human_heading = np.array(np.pi/3)
        else:
            human_position = np.array([random.randrange(100, 300), random.randrange(300, 600)])
            human_heading = np.random.uniform(low=-np.pi/3, high=np.pi/3)
        self.human_state = np.array([human_position[0], human_position[1], human_heading], dtype=np.float32)
        self.human_tripoints = np.array([[0, 25], [-10, -25], [10, -25]])


        while(np.min(wall_set_distance(self.walls, self.human_state)) <= 0):
            self.human_state = np.array([random.randrange(1, 400), random.randrange(1, 900)])

        # Goals
        self.robot_goal_rect = np.array([100, 300, 200, 300])
        self.human_goal_rect = np.array([1300, 300, 200, 300])
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

        self.reward = - np.linalg.norm(self.robot_state[:2] - self.robot_goal_rect[:2]) - np.linalg.norm(self.human_state[:2] - self.human_goal_rect[:2])
        self.reward = self.reward / 1000
        #print(self.reward)
        self.prev_reward = self.reward
        self.cumulative_reward = 0

        # Don't allow collisions, but allow the robot to turn around.
        wall_coord = self.walls[self.intent]
        wall_left = wall_coord[0]
        wall_right = WALL_XLEN
        wall_up = wall_coord[1] - WALL_YLEN
        wall_down = wall_up + WALL_YLEN
        rect = (wall_left, wall_up, WALL_XLEN, WALL_YLEN)

        if self.rgb_observation:
            self.get_image(resolution_scale=1)
            # from PIL import Image
            # img = Image.fromarray(self.img, 'RGB')
            # img.save('try.png')
            # import time
            # print("something")
            # time.sleep(10)

            observation = cv2.resize(self.img, (256, 256))
        else:
            human_delta = self.robot_state - self.human_state
            human_delta_pos = np.linalg.norm(human_delta[:2])
            human_delta_bearing = np.arctan2(human_delta[0], human_delta[1])
            human_wall_dist = wall_set_distance(self.walls, self.human_state)
            robot_wall_dist = wall_set_distance(self.walls, self.robot_state)
            self.dist_robot, _ = distance_to_goal(self.robot_state, self.robot_goal_rect)
            self.dist_human, _ = distance_to_goal(self.human_state, self.human_goal_rect)
            observation = np.concatenate((np.array([human_delta_pos, human_delta_bearing*100]),
                                         human_wall_dist, robot_wall_dist,
                                         wall_set_distance([rect], self.human_state),
                                         wall_set_distance([rect], self.robot_state),
                                         self.robot_state[-1:]*100, self.human_state[-1:]*100,
                                         np.array([self.dist_robot, self.dist_human])))
            observation /= 100
        #print(observation.shape)
        # observation = np.concatenate((self.robot_state, self.human_state))
        observation = {"obs": observation, "mode": np.eye(5)[self.intent]}


        return observation, {}

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
        if collision_with_boundaries(new_state) or violated_dist: #or intent_violation(new_state):
            new_state_x = np.array([xnew, y])
            new_state_y = np.array([x, ynew])
            wall_dist_x = wall_set_distance(self.walls, new_state_x)
            violated_dist_x = any(wall_dist_x <= 0)
            wall_dist_y = wall_set_distance(self.walls, new_state_y)
            violated_dist_y = any(wall_dist_y <= 0)
            if not(collision_with_boundaries(new_state_x) or violated_dist_x):# or intent_violation(new_state_x)):
                ynew = y
            elif not(collision_with_boundaries(new_state_y) or violated_dist_y):# or intent_violation(new_state_y)):
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

    def get_image(self, resolution_scale):
        robot_tripoints = state_to_tripoints(self.robot_state, self.robot_tripoints)*resolution_scale
        human_tripoints = state_to_tripoints(self.human_state, self.human_tripoints)*resolution_scale

        self.img = 255 - np.zeros((900*resolution_scale, 1600*resolution_scale, 3), dtype='uint8')

        # Display Goal
        cv2.rectangle(self.img, (self.robot_goal_rect[0]*resolution_scale, self.robot_goal_rect[1]*resolution_scale),
                      (self.robot_goal_rect[0]*resolution_scale + self.robot_goal_rect[2]*resolution_scale,
                       self.robot_goal_rect[1]*resolution_scale + self.robot_goal_rect[3]*resolution_scale),
                      (255, 0, 0), 3)

        # Display Human Goal
        cv2.rectangle(self.img, (self.human_goal_rect[0]*resolution_scale, self.human_goal_rect[1]*resolution_scale),
                      (self.human_goal_rect[0]*resolution_scale + self.human_goal_rect[2]*resolution_scale,
                       self.human_goal_rect[1]*resolution_scale + self.human_goal_rect[3]*resolution_scale),
                      (0, 0, 255), 3)

        # Display Robot Boundary
        cv2.circle(self.img, (int(self.robot_state[0]*resolution_scale), int(self.robot_state[1]*resolution_scale)), 50, (255, 0, 0), thickness=1, lineType=8, shift=0)

        # Display Human Boundary
        cv2.circle(self.img, (int(self.human_state[0]*resolution_scale), int(self.human_state[1]*resolution_scale)), 50, (0, 0, 255), thickness=1, lineType=8, shift=0)


        # Display Human Boundary

        for wall in self.walls[:-1]:
            cv2.rectangle(self.img, (wall[0]*resolution_scale, wall[1]*resolution_scale),
                          (wall[0]*resolution_scale + WALL_XLEN*resolution_scale,
                           wall[1]*resolution_scale + WALL_YLEN*resolution_scale), (0, 0, 0), -1)

        intent_wall = self.walls[self.intent]
        x, y, w, h = (intent_wall[0]*resolution_scale, intent_wall[1]*resolution_scale - WALL_YLEN*resolution_scale,
                      WALL_XLEN*resolution_scale, WALL_YLEN*resolution_scale)
        sub_img = self.img[y:y + h, x:x + w]
        intent_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        intent_rect[:, :, -1] = 255

        res = cv2.addWeighted(sub_img, 0.5, intent_rect, 0.5, 1.0)

        # # Putting the image back to its position
        self.img[y:y + h, x:x + w] = res

        # Display human and robot
        cv2.fillPoly(self.img, [human_tripoints.reshape(-1, 1, 2).astype(np.int32)], color=(0, 0, 255))
        cv2.fillPoly(self.img, [robot_tripoints.reshape(-1, 1, 2).astype(np.int32)], color=(255, 0, 0))

        cv2.waitKey(1)

    def render(self, resolution_scale=1):

        self.get_image(resolution_scale)

        display_img = cv2.flip(self.img, 1)

        self.img = cv2.flip(self.img, 1)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        if self.learning_agent == LearningAgent.HUMAN:
            la_str = "Red"
        else:
            la_str = "Blue"
        str = f"Learning agent: Both"
        cv2.putText(display_img, str, (1300*resolution_scale, 75*resolution_scale),
                    self.font, 0.75*resolution_scale, (0, 0, 0), 2, cv2.LINE_AA)

        for img in [self.img, display_img]:

            # Display the policy and intent
            mode = self.intent
            str = f"Human intent: hallway {mode}"
            cv2.putText(img, str, (1300 * resolution_scale, 25 * resolution_scale),
                        self.font, 0.75 * resolution_scale, (0, 0, 0), 2, cv2.LINE_AA)

            str = f"Cumulative reward: {self.cumulative_reward}"
            cv2.putText(img, str, (1300 * resolution_scale, 50 * resolution_scale),
                        self.font, 0.75 * resolution_scale, (0, 0, 0), 2, cv2.LINE_AA)

            for i, wall in enumerate(self.walls):
                strx = int(wall[0] + WALL_XLEN/2)
                stry = int(wall[1] - WALL_YLEN/2)
                str = f"{i}"
                cv2.putText(img, str, (strx*resolution_scale, stry*resolution_scale),
                            self.font, 0.75*resolution_scale, (0, 0, 0), 2, cv2.LINE_AA)

        if self.display_render:
            # self.img = cv2.resize(self.img, (960, 540))
            cv2.imshow('Hallway Environment', display_img)
            cv2.waitKey(1)

        # Takes step after fixed time
        t_end = time.time() + 0.05
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue
        return self.img
        
