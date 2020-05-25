import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath('../../.'))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model.SparseNet import SparseNet
from src.model.ImageDataset import NatPatchDataset
from src.utils.cmd_line import parse_args
from src.scripts.plotting import plot_rf

arg = parse_args()

params = f"kern={arg.kernel_size}_stride={arg.stride}_rlr={arg.r_learning_rate}_lr={arg.learning_rate}_lmda={arg.reg}"
checkpoint_path = f'../../trained_models/{params}/'
# create the checkpoint directory if it doesn't exist
Path(checkpoint_path).mkdir(exist_ok=True)

# save to tensorboard
board = SummaryWriter("../../runs/sparse-net/" + params)
arg = parse_args()
# if use cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create net
sparse_net = SparseNet(arg.n_neuron, arg.size, arg.kernel_size, arg.stride, R_lr=arg.r_learning_rate, lmda=arg.reg, device=device)
# load data
dataloader = DataLoader(NatPatchDataset(arg.batch_size, arg.size, arg.size), batch_size=250)
# train
optim = torch.optim.SGD([{'params': [sparse_net.U.weight,
    #sparse_net.conv_trans.weight
    ], "lr": arg.learning_rate}])
for e in range(arg.epoch):
    running_loss = 0
    c = 0
    for img_batch in tqdm(dataloader, desc='training', total=len(dataloader)):
        img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)
        #print('img batch shape: ')
        #print(img_batch.shape)
        # update
        pred = sparse_net(img_batch)
        loss = ((img_batch - pred) ** 2).sum()
        running_loss += loss.item()
        loss.backward()
        # update U
        optim.step()
        # zero grad
        sparse_net.zero_grad()
        # norm
        sparse_net.normalize_weights()
        c += 1
    board.add_scalar('General loss', running_loss / c, e * len(dataloader) + c)
    board.add_scalar('ista loss', sparse_net.get_ista_loss(), e * len(dataloader) + c)
    if e % 5 == 4:
        # plotting
        conv_out_width = (arg.size - 1) * arg.stride + (arg.kernel_size-1) + 1
        fig = plot_rf(sparse_net.U.weight.T.reshape(conv_out_width ** 2, arg.size, arg.size).cpu().data.numpy(), conv_out_width ** 2, arg.size)
        board.add_figure('RF', fig, global_step=e * len(dataloader) + c)
    if e % 10 == 9:
        # save checkpoint
        torch.save(sparse_net, f"../../trained_models/{params}/ckpt-{e+1}.pth")
torch.save(sparse_net, f"../../trained_models/{params}/ckpt-{e+1}.pth")
