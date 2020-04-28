import sys
sys.path.insert(0, '../..')

import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from src.model.ImageDataset import NatPatchDataset

def generate_image(fpath='../../data/IMAGES.mat'):
    size = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(NatPatchDataset(1, size, size), batch_size=1)
    model = torch.load('../../trained_models/kern=4_stride=2_rlr=0.001_lr=0.0007_lmda=0.0002/ckpt-90.pth', map_location=device)
    model.eval()

    for img_batch in dataloader:
        img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)
        pred = model(img_batch)
        plt.subplot(1, 2, 1)
        plt.imshow(img_batch.reshape(10, 10).cpu().numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(pred.reshape(10, 10).detach().cpu().numpy())
        plt.savefig('kern4_stride2.png',  cmap='Greys')
        return

generate_image()
