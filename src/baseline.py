from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class SketchyDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_names = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = self.root_dir + '/' + self.image_names[idx]
        sample = io.imread(image_path)

        if self.transform:
            sample = self.transform(sample)

        return sample

sketch_dataset = SketchyDataset(root_dir='../data/sketchy_data/sketch/tx_000000000000/tree')
photo_dataset = SketchyDataset(root_dir='../data/sketchy_data/photo/tx_000000000000/tree')


fig = plt.figure()

for i in range(len(photo_dataset)):
    sample = photo_dataset[i]

    
    plt.imshow(sample)
    plt.pause(1)

plt.close()

# augmentations

# pix2pix model
#   as baseline


