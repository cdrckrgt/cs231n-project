import argparse
import os
import numpy as np
import math
import scipy
import itertools
from skimage import io, transform
from PIL import Image

from torchvision import transforms, utils

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


nb_epochs = 10
batch_size = 4
lr = 1e-3
b1 = 0.5
b2 = 0.999
latent_dim = 100
sample_interval = 400

img_shape = (256, 256, 3) # i think
img_size = 256
channels = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class CoupledGenerators(nn.Module):
    def __init__(self):
        super(CoupledGenerators, self).__init__()

        self.init_size = img_size // 4
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.shared_conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
        )
        self.G1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.G2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)
        img1 = self.G1(img_emb)
        img2 = self.G2(img_emb)
        return img1, img2


class CoupledDiscriminators(nn.Module):
    def __init__(self):
        super(CoupledDiscriminators, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block

        self.shared_conv = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.D1 = nn.Linear(128 * ds_size ** 2, 1)
        self.D2 = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img1, img2):
        # Determine validity of first image
        out = self.shared_conv(img1)
        out = out.view(out.shape[0], -1)
        validity1 = self.D1(out)
        # Determine validity of second image
        out = self.shared_conv(img2)
        out = out.view(out.shape[0], -1)
        validity2 = self.D2(out)

        return validity1, validity2


# Loss function
adversarial_loss = torch.nn.MSELoss()

# Initialize models
coupled_generators = CoupledGenerators()
coupled_discriminators = CoupledDiscriminators()

coupled_generators.to(device)
coupled_discriminators.to(device)

# Initialize weights
coupled_generators.apply(weights_init_normal)
coupled_discriminators.apply(weights_init_normal)

# Configure data loader

class SketchyDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_names = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = self.root_dir + '/' + self.image_names[idx]
        sample = Image.open(image_path)

        if self.transform:
            sample = self.transform(sample)

        sample = np.array(sample).reshape(3, 256, 256)
        return sample

transform = transforms.Compose([
    transforms.RandomRotation(180)
])

sketch_dataset = SketchyDataset(root_dir='../data/tree_sketches', transform=transform)
photo_dataset = SketchyDataset(root_dir='../data/tree_photos', transform=transform)

dataloader1 = torch.utils.data.DataLoader(
    sketch_dataset,
    batch_size=batch_size,
    shuffle=True,
)

dataloader2 = torch.utils.data.DataLoader(
    photo_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(coupled_generators.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(coupled_discriminators.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(nb_epochs):
    for i, (imgs1, imgs2) in enumerate(zip(dataloader1, dataloader2)):

        batch_size = imgs1.shape[0]

        # Adversarial ground truths
        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        imgs1 = Variable(imgs1.type(Tensor).expand(imgs1.size(0), 3, img_size, img_size))
        imgs2 = Variable(imgs2.type(Tensor))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim))))

        # Generate a batch of images
        gen_imgs1, gen_imgs2 = coupled_generators(z)
        # Determine validity of generated images
        validity1, validity2 = coupled_discriminators(gen_imgs1, gen_imgs2)

        g_loss = (adversarial_loss(validity1, valid) + adversarial_loss(validity2, valid)) / 2

        g_loss.backward()
        optimizer_G.step()

        # ----------------------
        #  Train Discriminators
        # ----------------------

        optimizer_D.zero_grad()

        # Determine validity of real and generated images
        validity1_real, validity2_real = coupled_discriminators(imgs1, imgs2)
        validity1_fake, validity2_fake = coupled_discriminators(gen_imgs1.detach(), gen_imgs2.detach())

        d_loss = (
            adversarial_loss(validity1_real, valid)
            + adversarial_loss(validity1_fake, fake)
            + adversarial_loss(validity2_real, valid)
            + adversarial_loss(validity2_fake, fake)
        ) / 4

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, nb_epochs, i, len(dataloader1), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader1) + i
        if batches_done % sample_interval == 0:
            gen_imgs = torch.cat((gen_imgs1.data, gen_imgs2.data), 0)
            utils.save_image(gen_imgs, "../images/%d.png" % batches_done, nrow=8, normalize=True)


# this for inception score
# https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
