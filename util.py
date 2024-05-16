"""
GLNN code
------------------------------
Implementation of GLNN algorithm, which is proposed in the paper:
Robust Beamforming with Gradient-based Liquid Neural Network

References and Relevant Links
------------------------------
GitHub Repository:
https://github.com/tp1000d/GLNN

Related arXiv Paper:
https://arxiv.org/abs/2405.07291

file introduction
------------------------------
this is the utils file, including the initialization of the channel, the computation of the se and the loss, etc.

@author: X.Wang
"""
# <editor-fold desc="import package">
import numpy as np
import torch
import torch.nn as nn
import random
from ncps.wirings import AutoNCP
from ncps.torch import CfC, LTC
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import math
# </editor-fold>

# <editor-fold desc="define the constant">
tuningSteps = 3
optimizer_lr_w = 10e-3
nr_of_ue_antennas = 2 # N_k
nr_of_users = 4  # K
nr_of_BS_antennas = 64  # M

snr = 10
noise_power = 1
total_power = noise_power * 10 ** (snr / 10)

# we provide 2000 examples
nr_of_testing = 2000
# </editor-fold>


# <editor-fold desc="define the util functions">
def computeSE(H, W, sigma2, user_weights):
    # compute the SE and loss
    R = torch.zeros(nr_of_users)
    for k in range(nr_of_users):
        H_k = H[k, :, :]  # [2, 64]
        W_k = W[:, k].unsqueeze(-1)  # [64,1]
        # nominator
        A_k = H_k @ W_k @ W_k.conj().T @ H_k.conj().T  # [2, 2]
        # denominator
        B_k = H_k @ W @ W.conj().T @ H_k.conj().T - A_k + sigma2 * torch.eye(nr_of_ue_antennas)
        # SE for the user k. not weighted.
        # temp = torch.log2(torch.det(torch.eye(nr_of_ue_antennas) +A_k @ torch.inverse(B_k)).real)
        # since A_k and B_k are Hermitian square matrices, we can use the following formula to compute R_k to speed up.
        temp = torch.log2((torch.det(B_k + A_k) / torch.det(B_k)).real)
        R[k] = user_weights[k] * temp
    sum_rate = torch.sum(R)
    loss_fun = -sum_rate+torch.sum(torch.relu(2.5*torch.ones(nr_of_users)-R))*0.7+0.3*torch.std(R)**2
    return sum_rate, loss_fun


def init_X(ue_antenna_number, user_number, channel, power):
    # initialize the base matrix X and the beamforming matrix W
    X = (torch.randn(user_number*ue_antenna_number, user_number)
         + 1j * torch.randn(user_number*ue_antenna_number, user_number))
    W = channel.H@ X
    # normalize the collapsed beamforming vector X and the full beamforming vector W
    X = X / torch.norm(W) * math.sqrt(power)
    return X


def seed_everything(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)  # cpu random seed
    np.random.seed(seed)  # numpy random seed
    random.seed(seed)  # python random seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def NormalizeWX(H_stacked, X=None, W=None, power=0.1):
    # normalize the power.
    if W is None and X is None:
        raise ValueError('Both X and W are None')
    if W is None:
        W = H_stacked.H @ X
    normW = torch.norm(W)
    WW = math.sqrt(power) / torch.sqrt(normW)  # normalization coefficient
    X = X * WW  # normalize the compressed precoding matrix
    W = get_W(H_stacked, X)
    return W, X, normW


class MyDataset(Dataset):
    # data1 is the H, while data2 is for H_est
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]


def get_W(H_stacked, X):
    return H_stacked.H @ X


# </editor-fold>
