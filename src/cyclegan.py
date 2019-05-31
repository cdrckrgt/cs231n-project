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

# format is date.run_number this day
run = '053019.08'
if not os.path.exists('../saved_imgs/{}'.format(run)):
    os.mkdir('../saved_imgs/{}'.format(run))
if not os.path.exists('../weights/{}'.format(run)):
    os.mkdir('../weights/{}'.format(run))

writer = SummaryWriter('../logs/{}'.format(run))

batch_size = 64
nb_training_iterations = 10000
lr = 1e-4
betas = (0.5, 0.999)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

transform = transforms.Compose([
    transforms.Resize((256, 256)), # downsample the image, too large for efficient training
    transforms.ToTensor(),
    # transforms.Normalize((0, 0, 0), (0.3, 0.3, 0.3))
])

photo_data = datasets.ImageFolder('../data/sketchydata/photo/', transform)
sketch_data = datasets.ImageFolder('../data/sketchydata/sketch/', transform)

photo_loader = DataLoader(dataset=photo_data, batch_size=batch_size, shuffle=True)
sketch_loader = DataLoader(dataset=sketch_data, batch_size=batch_size, shuffle=True)

# create cycle generator
# encoder resnet decoder

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
        self.resnet4 = ResnetBlock(128)
        self.act7 = nn.LeakyReLU(0.2)
        self.resnet5 = ResnetBlock(128)
        self.act8 = nn.LeakyReLU(0.2)
        self.resnet6 = ResnetBlock(128)
        self.act9 = nn.LeakyReLU(0.2)
        self.resnet7 = ResnetBlock(128)
        self.act10 = nn.LeakyReLU(0.2)
        self.resnet8 = ResnetBlock(128)
        self.act11 = nn.LeakyReLU(0.2)
        self.resnet9 = ResnetBlock(128)
        self.act12 = nn.LeakyReLU(0.2)

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
        out = self.act7(self.resnet4(out))
        out = self.act8(self.resnet5(out))
        out = self.act9(self.resnet6(out))
        out = self.act10(self.resnet7(out))
        out = self.act11(self.resnet8(out))
        out = self.act12(self.resnet9(out))
        # decoder
        out = self.act13(self.bn4(self.convT1(out)))
        out = self.act14(self.bn5(self.convT2(out)))
        out = self.act15(self.convT3(out))
        return out


# create cycle discriminator
# normal dc discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(128, 1, 4, 2, 0)
        self.act4 = nn.Sigmoid()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.act3(self.bn3(self.conv3(out)))
        out = self.act4(self.conv4(out).squeeze())
        return out

# create generators and discriminators
G_XtoY = Generator().to(device)
G_YtoX = Generator().to(device)

D_X = Discriminator().to(device)
D_Y = Discriminator().to(device)


# create optimizers
G_opt = optim.Adam(list(G_XtoY.parameters()) + list(G_YtoX.parameters()), lr=lr, betas=betas)
D_opt = optim.Adam(list(D_X.parameters()) + list(D_Y.parameters()), lr=lr, betas=betas)
# D_opt = optim.SGD(list(D_X.parameters()) + list(D_Y.parameters()), lr=lr, momentum=0.9)

# training loop


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

def log_train_img(G_XtoY, G_YtoX, monitor_X, monitor_Y, i, save=False):
    fake_X = G_YtoX(monitor_Y)
    fake_Y = G_XtoY(monitor_X)

    X = monitor_X.cpu().data.numpy()
    fake_X = fake_X.cpu().data.numpy()
    Y = monitor_Y.cpu().data.numpy()
    fake_Y = fake_Y.cpu().data.numpy()

    merged = merge_images(X, fake_Y)
    writer.add_image('X, generated Y', merged.transpose(2, 0, 1), global_step=i)
    if save:
        path = '../saved_imgs/{}/iter_{}-X-Y.png'.format(run, i)
        scipy.misc.imsave(path, merged)
        print('Saved {}'.format(path))

    merged = merge_images(Y, fake_X)
    writer.add_image('Y, generated X', merged.transpose(2, 0, 1), global_step=i)
    if save:
        path = '../saved_imgs/{}/iter_{}-Y-X.png'.format(run, i)
        scipy.misc.imsave(path, merged)
        print('Saved {}'.format(path))

from torch.autograd import Variable

photo_iter = iter(photo_loader)
sketch_iter = iter(sketch_loader)


# monitoring batch
monitor_X, _ = photo_iter.next()
monitor_Y, _ = sketch_iter.next()
monitor_X = Variable(monitor_X).to(device)
monitor_Y = Variable(monitor_Y).to(device)


for i in range(nb_training_iterations):
    # load real images minibatch
    # compute discriminator losses for both domains for real images

    try :
        real_img_X, _ = photo_iter.next()
        real_img_Y, _ = sketch_iter.next()
    except StopIteration:
        photo_iter = iter(photo_loader)
        sketch_iter = iter(sketch_loader)
        photo_iter.next()
        sketch_iter.next()
        real_img_X, _ = photo_iter.next()
        real_img_Y, _ = sketch_iter.next()
        
    real_img_X = Variable(real_img_X).to(device)
    
    real_img_Y = Variable(real_img_Y).to(device)

    # generate fake images minibatch
    # compute discriminator losses for both domains for fake images

    # real loss
    D_opt.zero_grad()
    D_X_real_loss = torch.sum(torch.pow(D_X(real_img_X) - 1, 2)) / batch_size
    D_Y_real_loss = torch.sum(torch.pow(D_Y(real_img_Y) - 1, 2)) / batch_size

    D_real_loss = D_X_real_loss + D_Y_real_loss

    # cyclegan implementation does this to slow learning rate of D
    D_real_loss /= 2

    writer.add_scalar('D_X_real_loss', D_X_real_loss.item(), global_step=i)
    writer.add_scalar('D_Y_real_loss', D_Y_real_loss.item(), global_step=i)
    writer.add_scalar('D_real_loss', D_real_loss.item(), global_step=i)

    D_real_loss.backward()
    D_opt.step()

    # generating fake images for X and Y
    fake_img_X = G_YtoX(real_img_Y)
    fake_img_Y = G_XtoY(real_img_X)

    # fake loss
    D_opt.zero_grad()
    D_X_fake_loss = torch.sum(torch.pow(D_X(fake_img_X), 2)) / batch_size
    D_Y_fake_loss = torch.sum(torch.pow(D_Y(fake_img_Y), 2)) / batch_size

    D_fake_loss = D_X_fake_loss + D_Y_fake_loss

    # cyclegan implementation does this to slow learning rate of D
    D_fake_loss /= 2

    writer.add_scalar('D_X_fake_loss', D_X_fake_loss.item(), global_step=i)
    writer.add_scalar('D_Y_fake_loss', D_Y_fake_loss.item(), global_step=i)
    writer.add_scalar('D_fake_loss', D_fake_loss.item(), global_step=i)

    D_fake_loss.backward()
    D_opt.step()

    # cycle consistency loss
    G_opt.zero_grad()
        
    # generating fake images for X
    fake_img_X = G_YtoX(real_img_Y)

    # generator loss
    G_YtoX_loss = torch.sum(torch.pow(D_X(fake_img_X) - 1, 2)) / batch_size

    # reconstruct Y
    reconstructed_Y = G_XtoY(fake_img_X)

    # cycle consistency loss
    G_YtoXtoY_loss = torch.sum(torch.pow(real_img_Y - reconstructed_Y, 2)) / batch_size

    G_Y_loss = G_YtoX_loss + G_YtoXtoY_loss

    writer.add_scalar('G_Y_loss', G_Y_loss.item(), global_step=i)

    G_Y_loss.backward()
    G_opt.step()
    
    # cycle consistency loss
    G_opt.zero_grad()

    # generating fake images for Y
    fake_img_Y = G_XtoY(real_img_X)

    # generator loss
    G_XtoY_loss = torch.sum(torch.pow(D_Y(fake_img_Y) - 1, 2)) / batch_size

    # reconstruct X
    reconstructed_X = G_YtoX(fake_img_Y)

    # cycle consistency loss
    G_XtoYtoX_loss = torch.sum(torch.pow(real_img_X - reconstructed_X, 2)) / batch_size

    G_X_loss = G_XtoY_loss + G_XtoYtoX_loss

    writer.add_scalar('G_X_loss', G_X_loss.item(), global_step=i)

    G_X_loss.backward()
    G_opt.step()

    if i % 100 == 0:
        print('iter: ', i)
        print('D_real_loss: ', D_real_loss.item())
        print('D_fake_loss: ', D_fake_loss.item())
        print('G_X_loss: ', G_X_loss.item())
        print('G_Y_loss: ', G_Y_loss.item())
        log_train_img(G_XtoY, G_YtoX, monitor_X, monitor_Y, i)

    # model checkpointing
    if i != 0 and i % 500 == 0:
        torch.save(G_XtoY.state_dict(), '../weights/{0}/G_XtoY.iter_{1}.pt'.format(run, i))
        torch.save(G_YtoX.state_dict(), '../weights/{0}/G_YtoX.iter_{1}.pt'.format(run, i))
        torch.save(D_X.state_dict(), '../weights/{0}/D_X.iter_{1}.pt'.format(run, i))
        torch.save(D_Y.state_dict(), '../weights/{0}/D_Y.iter_{1}.pt'.format(run, i))
        # log_train_img(G_XtoY, G_YtoX, monitor_X, monitor_Y, i, save=True)

torch.save(G_XtoY.state_dict(), '../weights/{0}/G_XtoY.pt'.format(run))
torch.save(G_YtoX.state_dict(), '../weights/{0}/G_YtoX.pt'.format(run))
torch.save(D_X.state_dict(), '../weights/{0}/D_X.pt'.format(run))
torch.save(D_Y.state_dict(), '../weights/{0}/D_Y.pt'.format(run))

