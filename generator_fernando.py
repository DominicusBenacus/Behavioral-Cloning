import numpy as np
import skimage.transform as sktransform
import random
import sklearn
import cv2
import os
from random import shuffle

def generator_fernando(samples, batch_size=128):
    samples = shuffle(samples)
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0].strip())
                center_angle = float(batch_sample[1])

                # randomize brightness
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2HSV)
                random_brightness = .1 + np.random.uniform()
                center_image[:,:,2] = center_image[:,:,2] * random_brightness
                center_image = cv2.cvtColor(center_image, cv2.COLOR_HSV2RGB)

                # resize
                #center_image = cv2.resize(center_image, (img_height, img_width), interpolation=cv2.INTER_AREA)

                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)