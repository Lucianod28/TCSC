import sys
sys.path.insert(0, '../..')

import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from src.utils.cmd_line import parse_args
from src.model.ImageDataset import NatPatchDataset
from src.scripts.plotting import plot_rf

arg = parse_args()
fpath='../../data/IMAGES.mat'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(NatPatchDataset(1, arg.size, arg.size), batch_size=1)
model = torch.load(f"../../trained_models/kern={arg.kernel_size}_stride={arg.stride}_rlr="
f"{arg.r_learning_rate}_lr={arg.learning_rate}_lmda={arg.reg}/ckpt-100.pth", map_location=device)
model.eval()

conv_out_width = (arg.size - 1) * arg.stride + (arg.kernel_size-1) + 1

for img_batch in dataloader:
    # Actual image
    #img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)

    # Horizontal line
    #img_batch = torch.zeros((10, 10))
    #img_batch[4:6, :] = 1
    #img_batch = img_batch.reshape(1, 100).to(device)

    # Diagonal image
    img_batch = torch.ones((10))
    img_batch = torch.diag(img_batch) + torch.diag(img_batch[1:], -1) + torch.diag(img_batch[1:], 1)
    img_batch = img_batch.flip(0).reshape(1, 100).to(device)
    pred = model(img_batch)
    #activations = model.conv_trans(model.R).reshape(img_batch.shape[0], -1).reshape(-1)
    activations = model.R.reshape(400, 100)
    activations = None
    #activations = activations - activations.min()
    #activations = (activations / activations.max()) / 2
    fig = plot_rf(model.conv_trans.weight.T.reshape(-1, arg.kernel_size, arg.kernel_size).cpu().data.numpy(), arg.n_neuron, arg.kernel_size, alphas=activations)
    fig.savefig('activation.png')

    plt.subplot(1, 2, 1)
    plt.imshow(img_batch.reshape(10, 10).cpu().numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(pred.reshape(10, 10).detach().cpu().numpy(), cmap='gray')
    plt.savefig('reconstruction.png')
    break


