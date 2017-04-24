import numpy as np
import random
import matplotlib.image as mpimg
import os
import cv2
from keras.backend import tf as ktf


def resize_normalize(image):
    """
    Applies preprocessing pipeline to an image: crops `top` and `bottom`
    portions of image, resizes to 66*200 px and scales pixel values to [0, 1].
    """
    # resize
    #image = cv2.resize(image, (66, 200)) #first try
    resized = ktf.image.resize_images(image, (66, 200))
    #normalize
    image = image/255.0 - 0.5

    return image
