import numpy as np
import random
import matplotlib.image as mpimg
import os
import cv2

def resize_normalize(image, crop_top=.375,crop_bottom=.125):
    """
    Applies preprocessing pipeline to an image: crops `offset` and `bottom_offset`
    portions of image, resizes to 32x128 px and scales pixel values to [0, 1].
    """
    # resize
    image = cv2.resize(image, (66, 200))

    #normalize
    image = (image / 255.0) - 0.5

    return image