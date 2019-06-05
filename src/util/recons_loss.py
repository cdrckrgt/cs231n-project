import os
from PIL import Image
import numpy as np

run = '060319.run03'
path = '../saved_imgs/{}'.format(run)
loss = 0.0

for i, impaths in enumerate(zip(os.listdir(path + '/Y'), os.listdir(path + '/recons_Y'))):
    impath1, impath2 = impaths
    im1 = Image.open(path + '/Y/' + impath1)
    im2 = Image.open(path + '/recons_Y/' + impath2)

    diff = np.sum(np.asarray(im1) - np.asarray(im2)) / np.prod(np.asarray(im1).shape)
    loss += diff


loss /= len(os.listdir(path + '/Y'))

print('reconstruction loss is: ', loss)
