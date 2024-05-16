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
this is the net file, which declares the learning network as shown in the paper.
note that the NNs are declared here and the optimization process is implemented in the main file.

@author: X.Wang
"""
# <editor-fold desc="import package">
from util import *
# </editor-fold>


# <editor-fold desc="meta learning network"
class GLNNOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GLNNOptimizer, self).__init__()
        wiring = AutoNCP(hidden_size, output_size)  # 16 units, 1 motor neuron
        self.model = CfC(input_size, wiring, proj_size=output_size)
        self.hx = None

    def forward(self, gradient):
        gradient, self.hx = self.model.forward(gradient, self.hx)
        self.hx = self.hx.detach()
        return gradient

    def clearHiddenState(self):
        self.hx = None
        return



# <editor-fold desc="build meta-learners">
def GradientbasedLearner(optimizee, user_weights, H, X, sigma2, retain_graph_flag=True):
    X_internal = X.clone().detach().requires_grad_(True)  # clone the precoding matrix
    sum_loss_w = 0  # record the accumulated loss
    H_stacked = H.view(-1, nr_of_BS_antennas)
    V = get_W(H_stacked, X_internal)
    _, loss = computeSE(H, V, sigma2, user_weights)
    loss.backward(retain_graph=retain_graph_flag)  # compute the gradient
    X_grad = X_internal.grad.clone().detach()  # clone the gradient
    # as pytorch can not process complex number, we have to split the real and imaginary parts and concatenate them
    X_update = optimizee(torch.cat((X_grad.real, X_grad.imag), dim=1))
    # print(X_update.shape)
    sum_loss_w += loss
    X_internal_update = X_update[:, :nr_of_users] + 1j * X_update[:, nr_of_users:]
    X_internal = X_internal + X_internal_update
    X_update.retain_grad()
    X_internal.retain_grad()
    return X_internal


def InitOptimizer(input_size, hidden_size, output_size, optimizee_type='GLNN', weight_decay=0.0, lr=optimizer_lr_w):
    if optimizee_type == 'GLNN':
        optimizer = GLNNOptimizer(input_size, hidden_size, output_size)
    # add other models here if you want
    else:
        raise ValueError('Invalid network type')
    updater = torch.optim.Adam(optimizer.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer, updater
# </editor-fold>


# <editor-fold desc="initialize the network and optimizer">
# nn parameters
manifoldInputSize = nr_of_users * 2
manifoldHiddenSize = 30
manifoldOutputSize = nr_of_users * 2
manifoldBatchSize = nr_of_ue_antennas * nr_of_users
# define your own params here if you don't want to use gradient based learning
# </editor-fold>
