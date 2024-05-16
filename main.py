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
this is the main function which can be run directly

@author: X.Wang
"""
import scipy.io as sio
import torch
from tqdm import tqdm
from net import *
from torch.utils.data import DataLoader
from itertools import islice
import time
seed_everything(1)

# <editor-fold desc="load channel">
CEE = '10dB'
NetType = 'GLNN'
H_t = sio.loadmat(f'CSIdyn{nr_of_BS_antennas}.mat')['HH']  # load the channel H, numpy format
H_t = torch.tensor(H_t)  # transforms from numpy to torch format
user_weights = torch.ones(nr_of_users)
HH_est = sio.loadmat(f'CSIdyn{nr_of_BS_antennas}.mat')[f'HH_est_{CEE}'] # load the channel H_est, numpy format
HH_est = torch.tensor(HH_est)
# create the dataset
dataset = MyDataset(H_t, HH_est)
train_loader = DataLoader(dataset, batch_size=nr_of_users, shuffle=False, num_workers=0)

noise_power = 1
total_power = noise_power * 10 ** (snr / 10)
inputSize = manifoldInputSize
hiddenSize = manifoldHiddenSize
outputSize = manifoldOutputSize
Learner = GradientbasedLearner
optimizer_w, adam_w = InitOptimizer(inputSize, hiddenSize, outputSize, optimizee_type=NetType)
testResults = torch.zeros(nr_of_testing)
# </editor-fold>

print(f'start for CEE {CEE} with net = {NetType}... please wait...')
# testing
start_time_ms = time.time()
for item_index, data in tqdm(enumerate(islice(train_loader, nr_of_testing))):
    H, H_est = data
    H_est_stacked = H_est.view(-1, nr_of_BS_antennas)
    X = init_X(nr_of_ue_antennas, nr_of_users, H_est_stacked, total_power)
    X_init = X.clone()
    for epoch in range(tuningSteps):
        X = Learner(optimizer_w, user_weights, H_est, X_init, noise_power)
        W, X, _ = NormalizeWX(H_est_stacked, X=X, power=total_power)
        _, loss = computeSE(H_est, W, noise_power, user_weights)
        adam_w.zero_grad()
        loss.backward()
        adam_w.step()
    SE, _ = computeSE(H, W, noise_power, user_weights)
    testResults[item_index] = SE.detach()
end_time_ms = time.time()
print(f'testing for CEE {CEE} finished. Time cost: {end_time_ms - start_time_ms} seconds')
print("Here are the average se in the dynamic test:", (torch.mean(testResults)).item())
sio.savemat(f'./{NetType}_points_{nr_of_testing}_snr_{snr}_tune_{tuningSteps}_cee_{CEE}_BS_{nr_of_BS_antennas}.mat', {
    'hiddenSize': manifoldHiddenSize,
    'testResults': testResults,
    'tuningSteps': tuningSteps,
    'NetType': NetType
})
