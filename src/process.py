import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os

def canny(image, sigma = 0.333):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 - sigma) * v))

    edged = cv2.Canny(image, lower, upper)

    return edged

for image_name in os.listdir('./tree_synset/'):
    print('processing image: ', image_name)
    image = np.asarray(Image.open('tree_synset/' + image_name))
    # image = canny(image)
    image = Image.fromarray(image)
    # image = ImageOps.invert(image)
    # image = image.convert('1')
    image.save('./tree_synset_raw/' + image_name[:-5] + '.png')

