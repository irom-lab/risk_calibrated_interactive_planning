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

    def __init__(self, root_dir, train_set_size=5, is_train=True, max_pred=100):
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
        if is_train:
            subdirs = subdirs[:train_set_size]
        else:
            subdirs = subdirs[train_set_size:]
        self.traj_dict = OrderedDict()
        self.file_names = {}
        i = 0
        for subdir in subdirs:
            file_path = os.path.join(root_dir, subdir)
            if not os.path.isfile(file_path):
                continue
            self.traj_dict[subdir] = pd.read_csv(file_path,on_bad_lines='skip')
            self.file_names[i] = subdir
            i += 1
        self.root_dir = root_dir

    def __len__(self):
        return len(list(self.traj_dict.keys()))

    def __getitem__(self, idx):

        filename = self.file_names[idx]
        rollout_data = self.traj_dict[filename]

        traj_len = len(rollout_data.index)
        frac_stop = np.random.randint(10) + 3
        Tstop = traj_len // frac_stop

        state_history = torch.Tensor(rollout_data.iloc[:Tstop, :-1].values).cuda()
        robot_state_gt = torch.Tensor(rollout_data.iloc[Tstop:Tstop+self.max_pred, 4:6].values).cuda()
        human_state_gt = torch.Tensor(rollout_data.iloc[Tstop:Tstop+self.max_pred, 1:3].values).cuda()
        intent_gt = torch.Tensor(rollout_data.iloc[Tstop:Tstop+self.max_pred, -1].values).cuda()

        ret_dict = {"state_history": state_history,
                    "robot_state_gt": robot_state_gt,
                    "human_state_gt": human_state_gt,
                    "intent_gt": intent_gt}

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
