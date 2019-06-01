'''
Cedrick Argueta, Kevin Wang
cedrick@cs.stanford.edu
kwang98@stanford.edu


Implementation of CycleGAN for use on a dataset involving sketches of trees and tree photos

'''

import os
import numpy as np
import math
import scipy
import itertools
from skimage import io, transform
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

from torchvision import transforms, utils
from torchvision import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset


################################################################################
# Creating folders for saving images and weights
################################################################################

# format is date.run_number this day
run = '053119.run05'
if not os.path.exists('../saved_imgs/{}'.format(run)):
    os.mkdir('../saved_imgs/{}'.format(run))
if not os.path.exists('../weights/{}'.format(run)):
    os.mkdir('../weights/{}'.format(run))

writer = SummaryWriter('../logs/{}'.format(run))

################################################################################
# Setting hyperparameters
################################################################################

batch_size = 16
nb_epochs = 200
lr = 1e-3
betas = (0.5, 0.999)
use_label_smoothing = True
lambda_ = 10

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

################################################################################
# Loading data from folder and creating DataLoaders
################################################################################

transform_sketch = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
transform_photo = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

photo_data = datasets.ImageFolder('../data/sketchydata/photo/', transform_photo)
sketch_data = datasets.ImageFolder('../data/sketchydata/sketch/', transform_sketch)

class DualDomainDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        super().__init__()
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, index):
        '''
        returns a tuple that represents images from domain A and B
        '''
        A_idx = index % len(self.datasetA)
        B_idx = index % len(self.datasetB)

        return (self.datasetA[A_idx], self.datasetB[B_idx])

    def __len__(self):
        return max(len(self.datasetA), len(self.datasetB))

train_data = DualDomainDataset(photo_data, sketch_data)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


################################################################################
# Defining Generator and Discriminator Architectures
################################################################################

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.conv(x)
        out = out + x
        return out

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
       
        # encoder 
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU(0.2)

        # resnet
        self.resnet1 = ResnetBlock(128)
        self.act4 = nn.LeakyReLU(0.2)
        self.resnet2 = ResnetBlock(128)
        self.act5 = nn.LeakyReLU(0.2)
        self.resnet3 = ResnetBlock(128)
        self.act6 = nn.LeakyReLU(0.2)
        
        # decoder
        self.convT1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.act13 = nn.LeakyReLU(0.2)
        self.convT2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.act14 = nn.LeakyReLU(0.2)
        self.convT3 = nn.ConvTranspose2d(32, 3, 4, 2, 1)
        self.act15 = nn.Tanh()

    def forward(self, x):
        # encoder
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.act3(self.bn3(self.conv3(out)))
        # resnet
        out = self.act4(self.resnet1(out))
        out = self.act5(self.resnet2(out))
        out = self.act6(self.resnet3(out))

        # decoder
        out = self.act13(self.bn4(self.convT1(out)))
        out = self.act14(self.bn5(self.convT2(out)))
        out = self.act15(self.convT3(out))
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(256, 1, 32, 1, 0)
        self.act5 = nn.Sigmoid()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x))) 
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.act3(self.bn3(self.conv3(out)))
        out = self.conv5(out).squeeze().unsqueeze(1)
        out = self.act5(out)
        return out

G_XtoY = Generator().to(device)
G_YtoX = Generator().to(device)

D_X = Discriminator().to(device)
D_Y = Discriminator().to(device)


G_opt = optim.Adam(list(G_XtoY.parameters()) + list(G_YtoX.parameters()), lr=lr, betas=betas)
D_opt = optim.Adam(list(D_X.parameters()) + list(D_Y.parameters()), lr=lr, betas=betas)


################################################################################
# Auxilliary functions for visualization in Tensorboard
################################################################################

def merge_images(sources, targets):
    """Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    _, _, h, w = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([3, row * h, row * w * 2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
        merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
    return merged.transpose(1, 2, 0)

def log_train_img(G_XtoY, G_YtoX, monitor_X, monitor_Y, steps_done, save=False):
    fake_X = G_YtoX(monitor_Y)
    fake_Y = G_XtoY(monitor_X)

    X = monitor_X.cpu().data.numpy()
    fake_X = fake_X.cpu().data.numpy()
    Y = monitor_Y.cpu().data.numpy()
    fake_Y = fake_Y.cpu().data.numpy()

    merged = merge_images(X, fake_Y)
    writer.add_image('X, generated Y', merged.transpose(2, 0, 1), global_step=steps_done)
    if save:
        path = '../saved_imgs/{}/iter_{}-X-Y.png'.format(run, i)
        scipy.misc.imsave(path, merged)
        print('Saved {}'.format(path))

    merged = merge_images(Y, fake_X)
    writer.add_image('Y, generated X', merged.transpose(2, 0, 1), global_step=steps_done)
    if save:
        path = '../saved_imgs/{}/iter_{}-Y-X.png'.format(run, i)
        scipy.misc.imsave(path, merged)
        print('Saved {}'.format(path))

################################################################################
# Training Loop
################################################################################

steps_done = 0

monitor_data = next(iter(train_loader))
monitor_X_data, monitor_Y_data = monitor_data
monitor_X, _ = monitor_X_data
monitor_Y, _ = monitor_Y_data

for i_epoch in range(nb_epochs):
    for i_batch, data in enumerate(train_loader):

        # unpacking data from loader
        X_data, Y_data = data

        real_img_X, _ = X_data
        real_img_Y, _ = Y_data

        real_img_X = real_img_X.to(device)
        real_img_Y = real_img_Y.to(device)

        assert real_img_X.shape == real_img_Y.shape, 'mismatch in data shape from dataloader'

        ########################################################################
        # Discriminator Loss
        ########################################################################
        D_opt.zero_grad()

        # label smoothing
        true_label = 1
        if use_label_smoothing:
            true_label = torch.tensor(np.random.uniform(low=0.7, high=1.2, size=(batch_size, 1))).to(device).float()

        D_X_real_loss = torch.sum(torch.pow(D_X(real_img_X) - true_label, 2)) / batch_size
        D_Y_real_loss = torch.sum(torch.pow(D_Y(real_img_Y) - true_label, 2)) / batch_size

        D_real_loss = D_X_real_loss + D_Y_real_loss

        # slow learning rate of discriminator by halving the loss
        D_real_loss /= 2

        D_real_loss.backward()
        D_opt.step()

        # generating fake images for X and Y
        fake_img_X = G_YtoX(real_img_Y)
        fake_img_Y = G_XtoY(real_img_X)

        # fake loss
        fake_label = 0
        if use_label_smoothing:
            fake_label = torch.tensor(np.random.uniform(low=0.0, high=0.3, size=(batch_size, 1))).to(device).float()

        D_opt.zero_grad()
        D_X_fake_loss = torch.sum(torch.pow(D_X(fake_img_X) - fake_label, 2)) / batch_size
        D_Y_fake_loss = torch.sum(torch.pow(D_Y(fake_img_Y) - fake_label, 2)) / batch_size

        D_fake_loss = D_X_fake_loss + D_Y_fake_loss

        # slow learning rate of discriminator by halving the loss
        D_fake_loss /= 2

        D_fake_loss.backward()
        D_opt.step()

        D_loss = D_real_loss + D_fake_loss

        # logging scalars
        writer.add_scalar('discriminator loss on real X', D_X_real_loss.item(), global_step=steps_done)
        writer.add_scalar('discriminator loss on real Y', D_Y_real_loss.item(), global_step=steps_done)
        writer.add_scalar('discriminator loss on fake X', D_X_fake_loss.item(), global_step=steps_done)
        writer.add_scalar('discriminator loss on fake Y', D_Y_fake_loss.item(), global_step=steps_done)
        writer.add_scalar('total discriminator loss on real images', D_real_loss.item(), global_step=steps_done)
        writer.add_scalar('total discriminator loss on fake images', D_fake_loss.item(), global_step=steps_done)
        writer.add_scalar('total discriminator loss', D_loss.item(), global_step=steps_done)

        ########################################################################
        # Generator Loss
        ########################################################################

        # Y to X to Y
        G_opt.zero_grad()
            
        fake_img_X = G_YtoX(real_img_Y)

        G_YtoX_loss = torch.sum(torch.pow(D_X(fake_img_X) - true_label, 2)) / batch_size

        reconstructed_Y = G_XtoY(fake_img_X)
        G_YtoXtoY_loss = torch.sum(torch.pow(real_img_Y - reconstructed_Y, 2)) / batch_size

        # generating fake images for X
        G_Y_loss = G_YtoX_loss + lambda_ * G_YtoXtoY_loss

        G_Y_loss.backward()
        G_opt.step()
       
        # X to Y to X 
        G_opt.zero_grad()

        fake_img_Y = G_XtoY(real_img_X)

        G_XtoY_loss = torch.sum(torch.pow(D_Y(fake_img_Y) - true_label, 2)) / batch_size

        reconstructed_X = G_YtoX(fake_img_Y)
        G_XtoYtoX_loss = torch.sum(torch.pow(real_img_X - reconstructed_X, 2)) / batch_size

        G_X_loss = G_XtoY_loss + lambda_ * G_XtoYtoX_loss

        G_X_loss.backward()
        G_opt.step()

        G_loss = G_X_loss + G_Y_loss
 
        # logging scalars
        writer.add_scalar('generator X to Y loss', G_XtoY_loss.item(), global_step=steps_done)
        writer.add_scalar('generator Y to X loss', G_YtoX_loss.item(), global_step=steps_done)
        writer.add_scalar('generator cycle consistency loss for X', G_XtoYtoX_loss.item(), global_step=steps_done)
        writer.add_scalar('generator cycle consistency loss for Y', G_YtoXtoY_loss.item(), global_step=steps_done)
        writer.add_scalar('total generator loss for X', G_X_loss.item(), global_step=steps_done)
        writer.add_scalar('total generator loss for Y', G_Y_loss.item(), global_step=steps_done)
        writer.add_scalar('total generator loss', G_loss.item(), global_step=steps_done)

        steps_done += 1
    ############################################################################
    # Logging and Model Checkpointing
    ############################################################################

    if i_epoch % 10 == 0:
        log_train_img(G_XtoY, G_YtoX, monitor_X, monitor_Y, steps_done)

    if i_epoch != 0 and i_epoch % 500 == 0:
        torch.save(G_XtoY.state_dict(), '../weights/{0}/G_XtoY.iter_{1}.pt'.format(run, i_epoch))
        torch.save(G_YtoX.state_dict(), '../weights/{0}/G_YtoX.iter_{1}.pt'.format(run, i_epoch))
        torch.save(D_X.state_dict(), '../weights/{0}/D_X.iter_{1}.pt'.format(run, i_epoch))
        torch.save(D_Y.state_dict(), '../weights/{0}/D_Y.iter_{1}.pt'.format(run, i_epoch))

################################################################################
# Saving Trained Model
################################################################################

torch.save(G_XtoY.state_dict(), '../weights/{0}/G_XtoY.pt'.format(run))
torch.save(G_YtoX.state_dict(), '../weights/{0}/G_YtoX.pt'.format(run))
torch.save(D_X.state_dict(), '../weights/{0}/D_X.pt'.format(run))
torch.save(D_Y.state_dict(), '../weights/{0}/D_Y.pt'.format(run))

