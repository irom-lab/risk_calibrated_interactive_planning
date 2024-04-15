import numpy as np
import torch

import numpy as np
from stable_baselines3 import PPO, SAC  # , MultiModalPPO
# from sb3_contrib import RecurrentPPO
from environments.hallway_env import HallwayEnv
from environments.make_vectorized_hallway_env import make_env, make_bullet_env
import os
import time
import pandas as pd
from PIL import Image
from environments.hallway_env import HumanIntent
import matplotlib.pyplot as plt
from environments.hallway_env import HallwayEnv, prompt
from environments.pybullet_hallway_env import BulletHallwayEnv, prompt
from os.path import expanduser
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook
from scipy.stats import binom, beta
from utils.visualization_utils import get_img_from_fig, add_colorbar
import torch
import matplotlib
from matplotlib.ticker import FormatStrFormatter

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1

def hoeffding_bentkus(risk_values, alpha_val=0.9, n=100):
    '''Compute hoeffding-bentkus concentration bound.
    risk_values: tensor of shape [L] (L is size of lambda set)
    alpha_vales: uniform bound for maximum risk
    n: sample set size

    returns: p-value for each lambda val
    '''
    sample_risk_mean = risk_values.mean(-1)
    alpha_val = torch.Tensor([alpha_val]).to(risk_values.device)

    max_alpha = torch.minimum(risk_values, alpha_val)
    ce = cross_entropy(max_alpha, alpha_val).clip(min=0)  # clip avoids p-val > 1 (p-val will still be bad)
    left_term = np.exp(-n * ce)

    x = np.ceil(n * sample_risk_mean)
    bin_cdf = binom.cdf(x, n, alpha_val)
    right_term = np.e * bin_cdf
    right_term = torch.Tensor(right_term).to(risk_values.device)

    hb_p_val = torch.minimum(left_term, right_term)
    return hb_p_val

def cross_entropy(a, b):
    return a * np.log(a/(b+0.001)+0.001) + (1-a) * np.log((1-a)/(1-b+0.001) + 0.001)

def knowno_test_eval(N=500, eps_coverage=0.02, delta=0.01):
    '''Computes valid finite-sample epsilon miscoverage value for the given delta (failure rate) value,
     not to exceed the target miscoverage value'''
    check_range = np.linspace(eps_coverage, eps_coverage/10, 10)
    # return eps_coverage # test!!!!
    delta = 0.01 # test!!!
    for e in check_range:
        v = np.floor((N + 1)*e)
        a = N + 1 - v
        b = v
        p_miscov = 1-beta.ppf(delta, a, b)
        if p_miscov < eps_coverage:
            return e
    return None

def get_knowno_epsilon_values(miscoverage_max=0.45, num_points=10, dataset_size=500):
    '''Computes range of miscoverage values in the range [0, miscoverage_max] '''
    knowno_coverage_range = np.linspace(0.01, miscoverage_max, num_points)
    indices = list(range(len(knowno_coverage_range)))
    test_eps_vals = [knowno_test_eval(eps_coverage=e, N=500) for e in knowno_coverage_range]
    test_eps = [(i, j, k) for (i,j,k) in zip(indices, knowno_coverage_range, test_eps_vals)]
    test_eps_vals = np.unique(test_eps_vals)
    return knowno_coverage_range, test_eps_vals