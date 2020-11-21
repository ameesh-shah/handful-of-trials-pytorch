from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from numpy import linalg as LA
from scipy.io import savemat

from DotmapUtils import get_required_argument
from optimizers import CEMOptimizer

from tqdm import trange

import torch

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ICM:
    """
    Intrinsic curiosity module
    """
    def __init__(self, policy):
        # Get the MPC object and its related functions during instantiation
        self.policy = policy
        
        # Discount factor
        self.gamma = 1
    
    def __call__(self, sample):
        obs = sample['obs']
        ac = sample['ac']
        reward = sample['reward']
        
        # predict next step
        next_obs = self.policy._predict_next_obs(obs, ac)
        
        # compute difference / intrinsic reward
        r_i = self.gamma * LA.norm(next_obs - obs)
        
        # add difference to reward
        reward += r_i
        
        # return sample
        return {
            'obs': obs,
            'ac': ac,
            'reward': reward
        }
    