import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from collections import OrderedDict

class IntentPredictionDataset(Dataset):
    """Face Landmarks dataset."""

    _habitat_robot_pos_index = 1
    _habitat_human_pos_index = 4
    _habitat_end_of_obs = -15
    _habitat_max_obs = 200

    _hallway_robot_pos_index = 17
    _hallway_human_pos_index = 20
    _hallway_end_of_obs = -3


    def __init__(self, root_dir, train_set_size=5, is_train=True, max_pred=100, debug=False, min_len=10, target_len=200,
                 max_in_set=None, use_habitat=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.max_pred = max_pred
        subdirs = sorted(os.listdir(root_dir))
        self.is_train = is_train
        self.min_len = min_len = target_len
        self.use_habitat = use_habitat
        if is_train:
            if debug:
                subdirs = subdirs[:100]
            else:
                subdirs = subdirs[:train_set_size]
        else:
            if debug:
                subdirs = subdirs[:100]
            else:
                subdirs = subdirs[train_set_size:]
        self.traj_dict = OrderedDict()
        self.target_len=target_len
        self.file_names = {}
        i = 0
        for subdir in subdirs:
            file_path = os.path.join(root_dir, subdir)
            if not os.path.isfile(file_path):
                continue
            if max_in_set is not None and i >= max_in_set:
                continue
            traj_data = pd.read_csv(file_path, on_bad_lines='skip')
            if self.valid_traj(traj_data):
                self.traj_dict[subdir] = traj_data
                self.file_names[i] = subdir
                i += 1
        self.root_dir = root_dir

    def valid_traj(self, traj_data):
        return len(traj_data.index) >= self.min_len


    def __len__(self):
        return len(list(self.traj_dict.keys()))

    def __getitem__(self, idx):

        filename = self.file_names[idx]
        rollout_data = self.traj_dict[filename]

        if self.use_habitat:
            robot_ind_start = self._habitat_robot_pos_index
            human_ind_start = self._habitat_human_pos_index
            end_of_obs = self._habitat_end_of_obs
        else:
            robot_ind_start = self._hallway_robot_pos_index
            human_ind_start = self._hallway_human_pos_index
            end_of_obs = self._hallway_end_of_obs

        traj_len = len(rollout_data.index)
        traj_stop = traj_len - self.max_pred # np.random.randint(low=self.min_len, high=traj_len-self.max_pred)
        Tstop = traj_stop
        obs_history = torch.Tensor(rollout_data.iloc[:Tstop, :end_of_obs].values).cuda()
        obs_full = torch.Tensor(rollout_data.iloc[:, :end_of_obs].values).cuda()

        if self.use_habitat:
            max_obs = torch.zeros((obs_history.shape[0], self._habitat_max_obs))
            max_obs[:, :obs_history.shape[1]] = obs_history
            max_obs_full = torch.zeros((obs_full.shape[0], self._habitat_max_obs))
            max_obs_full[:, :obs_full.shape[1]] = obs_full
            obs_full = max_obs_full
            obs_history = max_obs
            all_actions = torch.Tensor(rollout_data.iloc[:, -15:-1].values).cuda()
        else:
            all_actions = torch.Tensor(rollout_data.iloc[:, -3:-1].values).cuda() # optimal action only


        robot_ind_end = robot_ind_start + 2
        human_ind_end = human_ind_start + 2
        robot_state_gt = torch.Tensor(rollout_data.iloc[Tstop:Tstop+self.max_pred, robot_ind_start:robot_ind_end].values).cuda()
        human_state_gt = torch.Tensor(rollout_data.iloc[Tstop:Tstop+self.max_pred, human_ind_start:human_ind_end].values).cuda()
        human_state_history = torch.Tensor(rollout_data.iloc[:Tstop, human_ind_end].values).cuda()
        robot_full_traj = torch.Tensor(rollout_data.iloc[:, robot_ind_start:robot_ind_end].values).cuda()
        human_full_traj = torch.Tensor(rollout_data.iloc[:, human_ind_start:human_ind_end].values).cuda()
        intent_gt = torch.Tensor(rollout_data.iloc[Tstop:Tstop+self.max_pred, -1].values).cuda()
        intent_full = torch.Tensor(rollout_data.iloc[:, -1].values).cuda()

        ret_dict = {"obs_history": obs_history,
                    "human_state_history": human_state_history,
                    "obs_full":  obs_full,
                    "robot_state_gt": robot_state_gt,
                    "human_state_gt": human_state_gt,
                    "robot_full_traj": robot_full_traj,
                    "human_full_traj": human_full_traj,
                    "intent_gt": intent_gt,
                    "intent_full": intent_full,
                    "all_actions": all_actions}

        return ret_dict

def collate_fn(data_list):

    ret_keys = data_list[0].keys()

    ret_dict = {}
    for k in ret_keys:
        ret_dict[k] = []

    for i, d in enumerate(data_list):
        for k in ret_keys:
            ret_dict[k].append(d[k])

    for k in ret_keys:
        ret_dict[k] = torch.nn.utils.rnn.pad_sequence(ret_dict[k], batch_first=True)

    return ret_dict


if __name__ == "__main__":
    data_path = "/home/jlidard/PredictiveRL/logs/1702317472/rollouts"
    dataset = IntentPredictionDataset(data_path)
    data = dataset[0]
    print(data)

    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    for test_itm in dataloader:
        print(test_itm)
