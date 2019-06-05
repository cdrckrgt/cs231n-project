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
from PIL import Image

from torchvision import transforms, utils
from torchvision import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary


################################################################################
# Creating folders for saving images and weights
################################################################################

# format is date.run_number this day
run = '060419.run02'
if not os.path.exists('../weights/{}'.format(run)):
    os.mkdir('../weights/{}'.format(run))
if not os.path.exists('../logs/{}'.format(run)):
    os.mkdir('../logs/{}'.format(run))

writer = SummaryWriter('../logs/{}'.format(run))


################################################################################
# Setting hyperparameters
################################################################################

batch_size = 8
nrows = 4
ncols = 2
nb_epochs = 200
lr = 2e-4
betas = (0.5, 0.999)
use_label_smoothing = True
lambda_ = 10.
use_noisy_labels = True
noisy_p = 0.05
should_halve_loss = True
buffer_max_size = 50
use_replay_buffer = True
# https://ssnl.github.io/better_cycles/report.pdf
use_better_cycles = True
gamma = 0.1
gamma_start = gamma
gamma_end = 0.95
lambda_start = lambda_
lambda_end = 0.0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

debug_weights = False


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

photo_data = datasets.ImageFolder('../data/palm_train/photo/', transform_photo)
sketch_data = datasets.ImageFolder('../data/palm_train/sketch/', transform_sketch)

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
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=32, pin_memory=True)


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
        self.up1 = nn.Upsample(scale_factor=2) # 256 x 256 x 256 => 256 x 512 x 512
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(256, 128, 3, 1, 0) # 256 x 512 x 512 => 128 x 256 x 256
        self.norm4 = nn.InstanceNorm2d(128)
        self.act13 = nn.ReLU()
        self.up2 = nn.Upsample(scale_factor=2) # 128 x 256 x 256 => 128 x 512 x 512
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 0) # 128 x 512 x 512 => 64 x 256 x 256
        self.norm5 = nn.InstanceNorm2d(64)
        self.act14 = nn.ReLU()
        self.pad4 = nn.ReflectionPad2d(3)
        self.conv6 = nn.Conv2d(64, 3, 7, 1, 0) # 64 x 256 x 256 => 3 x 256 x 256
        self.act15 = nn.Tanh()

    def forward(self, x):
        # encoder
        out = self.act1(self.norm1(self.conv1(self.pad1(x))))
        out = self.act2(self.norm2(self.conv2(out)))
        out = self.act3(self.norm3(self.conv3(out)))

        # transformer
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
        out = self.act13(self.norm4(self.conv4(self.pad2(self.up1(out)))))
        out = self.act14(self.norm5(self.conv5(self.pad3(self.up2(out)))))
        out = self.act15(self.conv6(self.pad4(out)))
        return out

class Discriminator(nn.Module):
    '''
    takes architecture cues from pix2pix discriminator
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1) # 3 x 256 x 256 => 64 x 128 x 128
        self.norm1 = nn.InstanceNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # 64 x 128 x 128 => 128 x 64 x 64
        self.norm2 = nn.InstanceNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # 128 x 64 x 64 => 256 x 32 x 32
        self.norm3 = nn.InstanceNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(256, 512, 4, 1, 1) # 256 x 32 x 32 => 512 x 31 x 31
        self.norm4 = nn.InstanceNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2)

        # apply a convolution to reduce depth to 1
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 1) # 512 x 31 x 31 => 1 x 30 x 30

    def forward(self, x):
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)))
        out = self.act3(self.norm3(self.conv3(out)))
        feature_out = self.act4(self.norm4(self.conv4(out)))
        out = self.conv5(feature_out) 
        return out, feature_out.detach()

def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        nn.init.normal_(layer.weight, mean=0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

G_XtoY = Generator().to(device)
G_YtoX = Generator().to(device)

D_X = Discriminator().to(device)
D_Y = Discriminator().to(device)

G_XtoY.apply(init_weights)
G_YtoX.apply(init_weights)
D_X.apply(init_weights)
D_Y.apply(init_weights)

G_opt = optim.Adam(list(G_XtoY.parameters()) + list(G_YtoX.parameters()), lr=lr, betas=betas)
D_opt = optim.Adam(list(D_X.parameters()) + list(D_Y.parameters()), lr=lr, betas=betas)

def lambda_rule(epoch):
    # lr_l = 1.0 - max(0, epoch + nb_epochs - 100) / float(100 + 1)
    lr_l = 1.0
    return lr_l
G_scheduler = LambdaLR(G_opt, lr_lambda=lambda_rule)
D_scheduler = LambdaLR(D_opt, lr_lambda=lambda_rule)

D_output_shape = (batch_size, 1, 30, 30)
D_criterion = nn.MSELoss()
G_criterion = nn.MSELoss()
cycle_criterion = nn.L1Loss()

summary(G_XtoY, (3, 256, 256))
summary(D_X, (3, 256, 256))


################################################################################
# History buffer for old generated iamges
################################################################################

class Buffer(object):
    '''
    based on: https://arxiv.org/pdf/1612.07828.pdf
    '''
    def __init__(self, max_size):
        self.max_size = max_size
        self.buf = []

    def push(self, batch):
        ret = []
        for i in range(batch_size):
            image = batch[i, :, :, :]
            if len(self.buf) < self.max_size:
                self.buf.append(image)
                ret.append(image)
            else:
                if np.random.rand() < 0.5:
                    idx = np.random.randint(self.max_size)
                    ret.append(self.buf[idx].clone())
                    self.buf[idx] = image
                else:
                    ret.append(image)
        return torch.stack(ret)


################################################################################
# Auxilliary functions 
################################################################################

def merge_images(sources, targets, row_x, row_y):
    _, _, h, w = sources.shape
    assert row_x * row_y == batch_size, 'incorrect number of rows or columns for tensorboard image creation'
    merged = np.zeros([3, row_x * h, row_y * w * 2])
    indices = [(i, j) for i in range(row_x) for j in range(row_y)]
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i, j = indices[idx]
        merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
        merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
    return merged

def denorm_for_print(image):
    image = image.cpu().data.numpy()
    image = (image * 0.5) + 0.5
    image = image.clip(0, 1)
    return image

def log_train_img(G_XtoY, G_YtoX, monitor_X, monitor_Y, steps_done, save=False):

    set_grad(G_XtoY, False)
    set_grad(G_YtoX, False)

    monitor_X = monitor_X.detach()
    monitor_Y = monitor_Y.detach()

    fake_X = G_YtoX(monitor_Y)
    fake_Y = G_XtoY(monitor_X)

    X = denorm_for_print(monitor_X)
    fake_Y = denorm_for_print(fake_Y)
    Y = denorm_for_print(monitor_Y)
    fake_X = denorm_for_print(fake_X)

    merged_XtoY = merge_images(X, fake_Y, nrows, ncols)
    writer.add_image('X, generated Y', merged_XtoY, global_step=steps_done)

    merged_YtoX = merge_images(Y, fake_X, nrows, ncols)
    writer.add_image('Y, generated X', merged_YtoX, global_step=steps_done)

    if save:
        if not os.path.exists('../saved_imgs/{}'.format(run)):
            os.mkdir('../saved_imgs/{}'.format(run))
        path = '../saved_imgs/{}/iter_{}-Y-X.png'.format(run, steps_done)
        scipy.misc.imsave(path, merged_YtoX)
        path = '../saved_imgs/{}/iter_{}-X-Y.png'.format(run, steps_done)
        scipy.misc.imsave(path, merged_XtoY)

def log_weights(model, which, steps_done):
    params = model.state_dict()
    for layer, weights in params.items():
        if 'conv' in layer:
            writer.add_histogram('{}.{}'.format(which, layer), weights, global_step=steps_done)

def set_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


################################################################################
# Training Loop
################################################################################

steps_done = 0

monitor_data = next(iter(train_loader))
monitor_X_data, monitor_Y_data = monitor_data

monitor_X, _ = monitor_X_data
monitor_Y, _ = monitor_Y_data

monitor_X = monitor_X.to(device)
monitor_Y = monitor_Y.to(device)

fake_X_history = Buffer(buffer_max_size)
fake_Y_history = Buffer(buffer_max_size)

from progress.bar import IncrementalBar

for i_epoch in range(nb_epochs):

    print('i_epoch', i_epoch)
    if use_better_cycles:
        writer.add_scalar('lambda', lambda_, global_step=steps_done)
        writer.add_scalar('gamma', gamma, global_step=steps_done)

    with IncrementalBar('batches', max=len(train_loader)) as bar:
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
            # Discriminator Loss
            ########################################################################

            set_grad(G_XtoY, False) # we save computation by setting these to no grad
            set_grad(G_YtoX, False)
            set_grad(D_X, True)
            set_grad(D_Y, True)

            # label smoothing adds noise to discriminator training, weakens D 
            true_label = torch.ones(D_output_shape).to(device)
            fake_label = torch.zeros(D_output_shape).to(device)
            if use_label_smoothing:
                true_label = torch.tensor(np.random.uniform(low=0.7, high=1.2, size=D_output_shape)).to(device).float()
                # fake_label = torch.tensor(np.random.uniform(low=0.0, high=0.3, size=D_output_shape)).to(device).float()
            if use_noisy_labels and np.random.rand() < noisy_p:
                true_label, fake_label = fake_label, true_label

            # real image loss
            D_opt.zero_grad()

            real_X_pred, real_X_features = D_X(real_img_X)
            real_Y_pred, real_Y_features = D_Y(real_img_Y)

            D_X_real_loss = D_criterion(real_X_pred, true_label)
            D_Y_real_loss = D_criterion(real_Y_pred, true_label)

            D_real_loss = D_X_real_loss + D_Y_real_loss

            if should_halve_loss:
                # slow learning rate of discriminator by halving the loss
                D_real_loss /= 2

            D_real_loss.backward()
            D_opt.step()

            # generating fake images
            fake_img_X = G_YtoX(real_img_Y)
            fake_img_Y = G_XtoY(real_img_X)

            if use_replay_buffer:
                fake_img_X = fake_X_history.push(fake_img_X)
                fake_img_Y = fake_Y_history.push(fake_img_Y)

                assert fake_img_X.shape == fake_img_Y.shape, 'mismatch in data shape from buffer'
                assert fake_img_X.shape == real_img_X.shape, 'incorrect shape returned from buffer'

            # fake image loss
            D_opt.zero_grad()

            fake_X_pred, _ = D_X(fake_img_X)
            fake_Y_pred, _ = D_Y(fake_img_Y)

            D_X_fake_loss = D_criterion(fake_X_pred, fake_label)
            D_Y_fake_loss = D_criterion(fake_Y_pred, fake_label)

            D_fake_loss = D_X_fake_loss + D_Y_fake_loss

            if should_halve_loss:
                # slow learning rate of discriminator by halving the loss
                D_fake_loss /= 2

            D_fake_loss.backward()
            D_opt.step()

            D_loss = D_real_loss + D_fake_loss

            if debug_weights:
                log_weights(D_X, 'D_X', steps_done)
                log_weights(D_Y, 'D_Y', steps_done)

            writer.add_scalar('discriminator loss on real X', D_X_real_loss.item(), global_step=steps_done)
            writer.add_scalar('discriminator loss on real Y', D_Y_real_loss.item(), global_step=steps_done)
            writer.add_scalar('discriminator loss on fake X', D_X_fake_loss.item(), global_step=steps_done)
            writer.add_scalar('discriminator loss on fake Y', D_Y_fake_loss.item(), global_step=steps_done)
            writer.add_scalar('total discriminator loss on real images', D_real_loss.item(), global_step=steps_done)
            writer.add_scalar('total discriminator loss on fake images', D_fake_loss.item(), global_step=steps_done)
            writer.add_scalar('total discriminator loss', D_loss.item(), global_step=steps_done)

            del D_X_real_loss
            del D_X_fake_loss
            del D_Y_real_loss
            del D_Y_fake_loss
            del D_real_loss
            del D_fake_loss
            del D_loss

            ########################################################################
            # Generator Loss
            ########################################################################

            set_grad(G_XtoY, True) # we save computation by setting these to no grad
            set_grad(G_YtoX, True)
            set_grad(D_X, False)
            set_grad(D_Y, False)

            true_label = torch.ones(D_output_shape).to(device)
            fake_label = torch.zeros(D_output_shape).to(device)

            # Y to X to Y
            G_opt.zero_grad()
 
            fake_img_X = G_YtoX(real_img_Y)

            fake_X_pred, _ = D_X(fake_img_X)
            G_YtoX_loss = G_criterion(fake_X_pred, true_label)

            reconstructed_Y = G_XtoY(fake_img_X)
            if use_better_cycles:
                _, reconstructed_Y_features = D_Y(reconstructed_Y)
                real_Y_pred = torch.sigmoid(real_Y_pred).detach()

                G_YtoXtoY_loss = torch.sum(real_Y_pred \
                    * (gamma * cycle_criterion(reconstructed_Y_features, real_Y_features) \
                    + (1 - gamma) * cycle_criterion(reconstructed_Y, real_img_Y))) / np.prod(D_output_shape)
            else:
                G_YtoXtoY_loss = cycle_criterion(real_img_Y, reconstructed_Y)

            G_Y_loss = G_YtoX_loss + lambda_ * G_YtoXtoY_loss

            G_Y_loss.backward()
            G_opt.step()
           
            # X to Y to X 
            G_opt.zero_grad()

            fake_img_Y = G_XtoY(real_img_X)

            fake_Y_pred, _ = D_Y(fake_img_Y)
            G_XtoY_loss = G_criterion(fake_Y_pred, true_label)

            reconstructed_X = G_YtoX(fake_img_Y)
            if use_better_cycles:
                _, reconstructed_X_features = D_X(reconstructed_X)
                real_X_pred = torch.sigmoid(real_X_pred).detach()

                G_XtoYtoX_loss = torch.sum(real_X_pred \
                    * (gamma * cycle_criterion(reconstructed_X_features, real_X_features) \
                    + (1 - gamma) * cycle_criterion(reconstructed_X, real_img_X))) / np.prod(D_output_shape)
            else:
                G_XtoYtoX_loss = cycle_criterion(real_img_X, reconstructed_X)

            G_X_loss = G_XtoY_loss + lambda_ * G_XtoYtoX_loss

            G_X_loss.backward()
            G_opt.step()

            G_loss = G_X_loss + G_Y_loss
     
            if debug_weights:
                log_weights(G_XtoY, 'G_XtoY', steps_done)
                log_weights(G_YtoX, 'G_YtoX', steps_done)

            writer.add_scalar('generator X to Y loss', G_XtoY_loss.item(), global_step=steps_done)
            writer.add_scalar('generator Y to X loss', G_YtoX_loss.item(), global_step=steps_done)
            writer.add_scalar('generator cycle consistency loss for X', G_XtoYtoX_loss.item(), global_step=steps_done)
            writer.add_scalar('generator cycle consistency loss for Y', G_YtoXtoY_loss.item(), global_step=steps_done)
            writer.add_scalar('total generator loss for X', G_X_loss.item(), global_step=steps_done)
            writer.add_scalar('total generator loss for Y', G_Y_loss.item(), global_step=steps_done)
            writer.add_scalar('total generator loss', G_loss.item(), global_step=steps_done)

            del G_XtoY_loss
            del G_YtoX_loss
            del G_XtoYtoX_loss
            del G_YtoXtoY_loss
            del G_X_loss
            del G_Y_loss
            del G_loss

            if steps_done % 50 == 0:
                log_train_img(G_XtoY, G_YtoX, monitor_X, monitor_Y, steps_done)

            steps_done += 1
            bar.next()

    ############################################################################
    # Gamma and Lambda and LR Scheduling
    ############################################################################

    if use_better_cycles:
        lambda_ -= (lambda_start - lambda_end) / nb_epochs
        gamma += (gamma_end - gamma_start) / nb_epochs

    G_scheduler.step()
    D_scheduler.step()

    ############################################################################
    # Model Checkpointing
    ############################################################################

    if i_epoch != 0 and i_epoch % 5 == 0:
        torch.save(G_XtoY.state_dict(), '../weights/{0}/G_XtoY.epoch{1}.pt'.format(run, i_epoch))
        torch.save(G_YtoX.state_dict(), '../weights/{0}/G_YtoX.epoch{1}.pt'.format(run, i_epoch))
        torch.save(D_X.state_dict(), '../weights/{0}/D_X.epoch{1}.pt'.format(run, i_epoch))
        torch.save(D_Y.state_dict(), '../weights/{0}/D_Y.epoch{1}.pt'.format(run, i_epoch))


################################################################################
# Saving Trained Model
################################################################################

torch.save(G_XtoY.state_dict(), '../weights/{0}/G_XtoY.pt'.format(run))
torch.save(G_YtoX.state_dict(), '../weights/{0}/G_YtoX.pt'.format(run))
torch.save(D_X.state_dict(), '../weights/{0}/D_X.pt'.format(run))
torch.save(D_Y.state_dict(), '../weights/{0}/D_Y.pt'.format(run))
