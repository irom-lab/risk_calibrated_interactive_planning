import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from collections import OrderedDict

import pickle

class IntentPredictionDataset(Dataset):
    """Face Landmarks dataset."""

    _habitat_robot_pos_index = 1
    _habitat_human_pos_index = 4
    _habitat_end_of_obs = -15
    _habitat_max_obs = 350
    _habitat_max_total_actions = 20

    _hallway_robot_pos_index = 17
    _hallway_human_pos_index = 20
    _hallway_end_of_obs = -3

    _vlm_image_dim = 100


    def __init__(self, root_dir, train_set_size=5, is_train=True, max_pred=100, debug=False, min_len=10, target_len=200,
                 max_in_set=None, use_habitat=False, use_vlm=False, use_bimanual=False,
                 calibration_set_size=0, seed=1234, is_calibration=False,
                 is_calibration_test=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.max_pred = max_pred
        subdirs = sorted(os.listdir(root_dir))
        num_total_data = len(subdirs)
        self.is_train = is_train
        self.min_len = min_len
        self.use_habitat = use_habitat
        self.use_vlm = use_vlm
        self.use_bimanual = use_bimanual
        indices = list(range(num_total_data))
        np.random.seed(seed)
        np.random.shuffle(indices)
        indices_shuffled = indices # should be the same for train, test, etc
        safety_factor = 10
        if is_train:
            indices = indices_shuffled[:train_set_size]
        elif is_calibration:
            indices = indices_shuffled[train_set_size+calibration_set_size:train_set_size+calibration_set_size+max_in_set+safety_factor]
        elif is_calibration_test:
            indices = indices_shuffled[train_set_size+calibration_set_size:train_set_size+calibration_set_size+max_in_set+safety_factor]
        else:
            indices = indices_shuffled[train_set_size:]
            # if not use_habitat and not use_vlm:
            #     indices = \
            #         [3989, 1162, 11973, 16980, 7484, 10368, 2543, 5905, 4407, 5894, 15760, 9397, 16761, 17519, 10212, 7936,
            #      1632, 11797, 2086, 2481, 4852, 16183, 6991, 17185, 14995, 13337, 10405, 15776, 7476, 13566, 828, 9542,
            #      2602, 3222, 1444, 15867, 12090, 14340, 15634, 8471, 11155, 6492, 8873, 16603, 15749, 6903, 4874, 5133,
            #      14669, 13083, 3595, 4807, 13460, 6134, 12429, 8506, 15927, 8409, 14633, 2751]

        subdirs = [subdirs[i] for i in indices]
        if is_calibration_test:
            print(subdirs)
        self.traj_dict = OrderedDict()
        self.target_len = min_len
        self.file_names = {}
        i = 0
        for subdir in subdirs[:max_in_set+20]:
            file_path = os.path.join(root_dir, subdir)
            if self.use_vlm:
                file_path = os.path.join(file_path, "ground_truth.csv")
            elif self.use_bimanual:
                file_path = os.path.join(file_path, "intent.pkl")
            if not os.path.isfile(file_path):
                continue
            if max_in_set is not None and i >= max_in_set:
                continue
            if not use_bimanual:
                traj_data = pd.read_csv(file_path, on_bad_lines='skip')
            else:
                traj_data = pickle.load(open(file_path, "rb"))
            if self.valid_traj(traj_data):
                full_path = os.path.join(root_dir, subdir)
                self.traj_dict[full_path] = traj_data
                self.file_names[i] = full_path
                i += 1
        self.root_dir = root_dir

    def valid_traj(self, traj_data):
        if self.use_vlm or self.use_bimanual:
            return True
        return len(traj_data.index) >= self.min_len


    def __len__(self):
        return len(list(self.traj_dict.keys()))

    def __getitem__(self, idx):

        filename = self.file_names[idx]
        rollout_data = self.traj_dict[filename]

        #TODO: clean up for cleaner handling of environments.

        if self.use_habitat:
            # hacky way to get variable num of intent
            temp_hab = torch.Tensor(rollout_data.iloc[:, :].values)
            num_intent = (torch.count_nonzero(temp_hab, 0) < torch.count_nonzero(temp_hab, 0).max() - 1).sum() // 2
            num_intent = num_intent.item()
            robot_ind_start = self._habitat_robot_pos_index
            human_ind_start = self._habitat_human_pos_index
            end_of_obs = -num_intent*2 - 1
        elif self.use_vlm or self.use_bimanual:
            ground_truth_intent = rollout_data["Groundtruth"]
        else:
            robot_ind_start = self._hallway_robot_pos_index
            human_ind_start = self._hallway_human_pos_index
            end_of_obs = self._hallway_end_of_obs


        if not self.use_vlm and not self.use_bimanual:
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
            all_actions = torch.zeros(rollout_data.shape[0], self._habitat_max_total_actions)
            all_actions[:, :num_intent*2] = torch.Tensor(rollout_data.iloc[:, end_of_obs:-1].values).cuda()
            all_actions = torch.clamp(all_actions, min=-10, max=10)
        elif self.use_vlm or self.use_bimanual:
            pass
        else:
            all_actions = torch.Tensor(rollout_data.iloc[:, -3:-1].values).cuda() # optimal action only
            num_intent = 5

        if self.use_vlm:
            ret_dict = {"directory_name": filename,
                        "intent_full": torch.Tensor(ground_truth_intent)}
        elif self.use_bimanual:
            scenario_num = filename.split('/')[-1].split('_')[-1]
            scenario_num = int(scenario_num)
            language_filename = os.path.join(filename, '..', '..')
            if scenario_num < 201:
                language_filename = os.path.join(language_filename, 'img_200.json')
            elif scenario_num < 401:
                language_filename = os.path.join(language_filename, 'img_400.json')
            elif scenario_num < 601:
                language_filename = os.path.join(language_filename, 'img_600.json')
            else:
                language_filename = os.path.join(language_filename, 'img_615.json')
            import json
            language_descriptions = json.load(open(language_filename, "rb"))
            desc = language_descriptions[str(scenario_num)]
            valid_intent = torch.zeros(5).cuda()
            intent_as_tensor = torch.Tensor([ground_truth_intent]).cuda()
            if isinstance(ground_truth_intent, int) or isinstance(ground_truth_intent, float):
                if ground_truth_intent > 5:
                    ground_truth_intent = 5
                valid_intent[int(ground_truth_intent)-1] = 1
            else:
                for i in range(len(ground_truth_intent)):
                    el = int(ground_truth_intent[i])
                    if el > 5:
                        el = 5
                    valid_intent[el-1] = 1
            ret_dict = {"directory_name": filename,
                        "instruction": desc,
                        "intent_full": valid_intent[None]}


        else:

            robot_ind_end = robot_ind_start + 3
            human_ind_end = human_ind_start + 3
            robot_state_gt = torch.Tensor(rollout_data.iloc[Tstop:Tstop+self.max_pred, robot_ind_start:robot_ind_end].values).cuda()
            human_state_gt = torch.Tensor(rollout_data.iloc[Tstop:Tstop+self.max_pred, human_ind_start:human_ind_end].values).cuda()
            human_state_history = torch.Tensor(rollout_data.iloc[:Tstop, human_ind_end].values).cuda()
            robot_full_traj = torch.Tensor(rollout_data.iloc[:, robot_ind_start:robot_ind_end].values).cuda()
            human_full_traj = torch.Tensor(rollout_data.iloc[:, human_ind_start:human_ind_end].values).cuda()
            intent_gt = torch.Tensor(rollout_data.iloc[Tstop:Tstop+self.max_pred, -1].values).cuda()
            intent_full = torch.Tensor(rollout_data.iloc[:, -1].values).cuda()

            ret_dict = {"directory_name": filename,
                        "obs_history": obs_history,
                        "human_state_history": human_state_history,
                        "obs_full":  obs_full,
                        "robot_state_gt": robot_state_gt,
                        "human_state_gt": human_state_gt,
                        "robot_full_traj": robot_full_traj,
                        "human_full_traj": human_full_traj,
                        "intent_gt": intent_gt,
                        "intent_full": intent_full,
                        "num_intent": num_intent * torch.ones(rollout_data.shape[0], 1),
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
        if type(ret_dict[k][0]) is torch.Tensor:
            ret_dict[k] = torch.nn.utils.rnn.pad_sequence(ret_dict[k], batch_first=True)

    ret_dict["batch_size"] = len(data_list)

    return ret_dict

def collate_fn_stack_only(data_list):

    ret_keys = data_list[0].keys()

    ret_dict = {}
    for k in ret_keys:
        ret_dict[k] = []

    for i, d in enumerate(data_list):
        for k in ret_keys:
            ret_dict[k].append(d[k])

    for k in ret_keys:
        if type(ret_dict[k][0]) is torch.Tensor:
            ret_dict[k] = torch.stack(ret_dict[k], 0)

    ret_dict["batch_size"] = len(data_list)


    return ret_dict


if __name__ == "__main__":
    data_path = "/home/jlidard/PredictiveRL/logs/1702317472/rollouts"
    dataset = IntentPredictionDataset(data_path)
    data = dataset[0]
    print(data)

    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    for test_itm in dataloader:
        print(test_itm)
