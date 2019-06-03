'''
Cedrick Argueta, Kevin Wang
cedrick@cs.stanford.edu
kwang98@stanford.edu

Testing file for CycleGAN implementation.
'''

import os
import numpy as np
import math
import scipy
import itertools
from skimage import io, transform
from PIL import Image

from torchvision import transforms, utils
from torchvision import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary


################################################################################
# Creating folders for saving images and weights
################################################################################

# format is date.run_number this day
run = '060219.run09'

################################################################################
# Setting hyperparameters
################################################################################

batch_size = 4
nrows = 2
ncols = 2

device = torch.device('cpu')
dtype = torch.FloatTensor


################################################################################
# Loading data from folder and creating DataLoaders
################################################################################

transform_sketch = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])
transform_photo = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(hue=0.05, saturation=0.05, brightness=0.05),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

photo_data = datasets.ImageFolder('../data/tu-berlin-trees/photo/', transform_photo)
sketch_data = datasets.ImageFolder('../data/tu-berlin-trees/sketch/', transform_sketch)

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
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=32, pin_memory=True)


################################################################################
# Defining Generator and Discriminator Architectures
################################################################################

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 0)
        self.norm1 = nn.InstanceNorm2d(dim)
        self.act1 = nn.ReLU()
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 0)
        self.norm2 = nn.InstanceNorm2d(dim)

    def forward(self, x):
        out = self.act1(self.norm1(self.conv1(self.pad1(x))))
        out = self.norm2(self.conv2(self.pad2(out)))
        out = out + x
        return out

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
 
        # encoder 
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, 64, 7, 1, 0) # 3 x 262 x 262 => 64 x 256 x 256
        self.norm1 = nn.InstanceNorm2d(64)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1) # 64 x 256 x 256 => 128 x 128 x 128, fractional striding
        self.norm2 = nn.InstanceNorm2d(128)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1) # 128 x 128 x 128 => 256 x 64 x 64, fractional striding
        self.norm3 = nn.InstanceNorm2d(256)
        self.act3 = nn.ReLU()

        # transformer
        self.resnet1 = ResnetBlock(256)
        self.resnet2 = ResnetBlock(256)
        self.resnet3 = ResnetBlock(256)
        self.resnet4 = ResnetBlock(256)
        self.resnet5 = ResnetBlock(256)
        self.resnet6 = ResnetBlock(256)
        self.resnet7 = ResnetBlock(256)
        self.resnet8 = ResnetBlock(256)
        self.resnet9 = ResnetBlock(256)
        
        # decoder
        # https://distill.pub/2016/deconv-checkerboard/
        self.convT1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1) # 256 x 64 x 64 => 128 x 128 x 128
        # self.up1 = nn.Upsample(scale_factor=2) # 256 x 256 x 256 => 256 x 512 x 512
        # self.pad2 = nn.ReflectionPad2d(1)
        # self.conv4 = nn.Conv2d(256, 128, 3, 1, 0) # 256 x 512 x 512 => 128 x 256 x 256
        self.norm4 = nn.InstanceNorm2d(128)
        self.act13 = nn.ReLU()
        self.convT2 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1) # 128 x 128 x 128 => 64 x 256 x 256
        # self.up2 = nn.Upsample(scale_factor=2) # 128 x 256 x 256 => 128 x 512 x 512
        # self.pad3 = nn.ReflectionPad2d(1)
        # self.conv5 = nn.Conv2d(128, 64, 3, 1, 0) # 128 x 512 x 512 => 64 x 256 x 256
        self.norm5 = nn.InstanceNorm2d(64)
        self.act14 = nn.ReLU()
        self.pad4 = nn.ReflectionPad2d(3)
        self.conv4 = nn.Conv2d(64, 3, 7, 1, 0) # 64 x 256 x 256 => 3 x 256 x 256
        self.act15 = nn.Tanh()

    def forward(self, x):
        # encoder
        out = self.act1(self.norm1(self.conv1(self.pad1(x))))
        out = self.act2(self.norm2(self.conv2(out)))
        out = self.act3(self.norm3(self.conv3(out)))

        # resnet
        out = self.resnet1(out)
        out = self.resnet2(out)
        out = self.resnet3(out)
        out = self.resnet4(out)
        out = self.resnet5(out)
        out = self.resnet6(out)
        out = self.resnet7(out)
        out = self.resnet8(out)
        out = self.resnet9(out)

        # decoder
        out = self.act13(self.norm4(self.convT1(out)))
        out = self.act14(self.norm5(self.convT2(out)))
        # out = self.act13(self.norm4(self.conv4(self.pad2(self.up1(x)))))
        # out = self.act14(self.norm5(self.conv5(self.pad3(self.up2(x)))))
        out = self.act15(self.conv4(self.pad4(out)))
        return out

G_XtoY = Generator()
G_XtoY.load_state_dict(torch.load('../weights/060219.run09/G_XtoY.epoch30.pt', map_location=device))
G_YtoX = Generator()
G_YtoX.load_state_dict(torch.load('../weights/060219.run09/G_YtoX.epoch30.pt', map_location=device))


################################################################################
# Auxilliary functions 
################################################################################

def merge_images(sources, targets, row_x, row_y):
    _, _, h, w = sources.shape
    assert row_x * row_y == batch_size, 'incorrect number of rows or columns for tensorboard image creation'
    merged = np.zeros([3, row_x * h, row_y * w * 2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row_x
        j = idx % row_y
        merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
        merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
    return merged

def denorm_for_print(image):
    image = image.cpu().data.numpy()
    image = (image * 0.5) + 0.5
    image = image.clip(0, 1)
    return image

def log_train_img(G_XtoY, G_YtoX, monitor_X, monitor_Y, steps_done):

    monitor_X = monitor_X.detach()
    monitor_Y = monitor_Y.detach()

    fake_X = G_YtoX(monitor_Y)
    fake_Y = G_XtoY(monitor_X)

    X = denorm_for_print(monitor_X)
    fake_Y = denorm_for_print(fake_Y)
    Y = denorm_for_print(monitor_Y)
    fake_X = denorm_for_print(fake_X)

    merged_XtoY = merge_images(X, fake_Y, nrows, ncols).transpose(1, 2, 0)

    merged_YtoX = merge_images(Y, fake_X, nrows, ncols).transpose(1, 2, 0)

    if not os.path.exists('../saved_imgs/{}'.format(run)):
        os.makedirs('../saved_imgs/{}'.format(run))
    path = '../saved_imgs/{}/iter_{}-Y-X.png'.format(run, steps_done)
    scipy.misc.imsave(path, merged_YtoX)
    path = '../saved_imgs/{}/iter_{}-X-Y.png'.format(run, steps_done)
    scipy.misc.imsave(path, merged_XtoY)

def set_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


################################################################################
# Evaluation Loop
################################################################################


for i_batch, data in enumerate(train_loader):

    # unpacking data from loader
    X_data, Y_data = data

    real_img_X, _ = X_data
    real_img_Y, _ = Y_data

    real_img_X = real_img_X.to(device)
    real_img_Y = real_img_Y.to(device)

    assert real_img_X.shape == real_img_Y.shape, 'mismatch in data shape from dataloader'
    assert real_img_X.shape[0] == batch_size, 'incorrect batch_size for loaded batch'

    ########################################################################
    # Converting and Saving
    ########################################################################

    set_grad(G_XtoY, False) # we save computation by setting these to no grad
    set_grad(G_YtoX, False)

    print('working on batch {}...'.format(i_batch))
    log_train_img(G_XtoY, G_YtoX, real_img_X, real_img_Y, i_batch)

