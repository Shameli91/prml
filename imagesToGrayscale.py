from PIL import Image
from skimage import color
from skimage import io
from skimage.io import imsave
import numpy as np
import os

for image in os.listdir('../pics/'):
    img = color.rgb2gray(io.imread('../pics/' + image))
    print(img.shape)
    #img = img.astype(np.uint8)
    #img = Image.open('../pics/' + image).convert('LA')
    #img = Image.fromarray(img)
    imsave('../picsgs/' + image, img)