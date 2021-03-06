import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseNet(nn.Module):

    def __init__(self, N:int, M:int, K:int, S:int, T:bool, R_lr:float, lmda:float=5e-3, device=None):
        super(SparseNet, self).__init__()
        self.N = N
        self.M = M
        self.K = K
        self.T = T
        self.R_lr = R_lr
        self.lmda = lmda
        # synaptic weights
        self.device = torch.device("cpu") if device is None else device
        conv_out_width = (self.M - 1) * S + (K-1) + 1
        self.U = nn.Linear(conv_out_width ** 2, self.M ** 2, bias=False).to(self.device)
        # responses
        self.R = None
        # transpose convolution
        self.conv_trans = nn.ConvTranspose2d(in_channels=self.N, out_channels=1,
                kernel_size=K, stride=S).to(self.device)

        # initialize weights with a constant
        #nn.init.constant_(self.conv_trans.weight.data,.2)
        self.normalize_weights()

    def ista_(self, img_batch):
        # create R
        self.R = torch.zeros((img_batch.shape[0], self.N, self.M, self.M),
                requires_grad=True, device=self.device)
        converged = False
        # update R
        optim = torch.optim.SGD([{'params': self.R, "lr": self.R_lr}])
        # train
        self.ista_loss = 0.
        last_change = None
        while not converged:
            old_R = self.R.clone().detach()
            # pred
            pred = self.U(self.conv_trans(self.R).reshape(img_batch.shape[0], -1))
            # loss
            loss = ((img_batch - pred) ** 2).sum()
            self.ista_loss += loss.item()
            loss.backward()
            # update R in place
            optim.step()
            # zero grad
            self.zero_grad()
            # prox
            self.R.data = SparseNet.soft_thresholding_(self.R, self.lmda)
            # convergence
            change = torch.norm(self.R - old_R) / torch.norm(old_R)
            if torch.isnan(change):  # convergence debugging
                if (last_change is None):
                    print(change.item())
                else:
                    print(last_change.item(), change.item())
                exit(1)
            last_change = change
            converged = change < 0.01

    @staticmethod
    def soft_thresholding_(x, alpha):
        with torch.no_grad():
            rtn = F.relu(x - alpha) - F.relu(-x - alpha)
        return rtn.data

    def zero_grad(self):
        self.R.grad.zero_()
        self.U.zero_grad()
        self.conv_trans.zero_grad()

    def normalize_weights(self):
        with torch.no_grad():
            self.U.weight.data = F.normalize(self.U.weight.data, dim=0)
            if self.T:  # if we're training conv_trans, normalize it
                self.conv_trans.weight.data = F.normalize(
                        self.conv_trans.weight.data.reshape(-1, 1, self.K**2), dim=2).reshape(-1, 1, self.K, self.K)

    def forward(self, img_batch):
        # first fit
        self.ista_(img_batch)
        # now predict again
        conv_transed = self.conv_trans(self.R).reshape(img_batch.shape[0], -1)
        pred = self.U(conv_transed)
        return pred

    def get_ista_loss(self):
        return self.ista_loss
