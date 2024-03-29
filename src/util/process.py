import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os

def canny(image, sigma = 0.333):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 - sigma) * v))

    lower = 235
    upper = 250

    edged = cv2.Canny(image, lower, upper)

    return edged

for image_name in os.listdir('./palm_synset/'):
    print('processing image: ', image_name)
    image = np.asarray(Image.open('palm_synset/' + image_name))
    image = cv2.bilateralFilter(image, 9, 160 ,160)
    image = canny(image)
    image = Image.fromarray(image)
    image = ImageOps.invert(image)
    image = image.convert('1')
    image.save('./palm_synset_edged/' + image_name[:-5] + '.png')

