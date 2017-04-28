import numpy as np
import skimage.transform as sktransform
import random
import cv2
import os
from random import shuffle


# set up some parameter for dealing with the different camera positions
cameras = ['left', 'center', 'right']
left_right_steering_correction = [.25, 0., -.25]
STEERING_ANGLE = 3


def generate_samples(data,rootPath, augment=True, batch_size = 128):
    """
    Keras generator yielding batches of training/validation data.
    Applies data augmentation pipeline if `augment` is True.

    Argument pipeline contains following augmentation prcesses
    - use images from all camera. Make the data set three times bigger
    -  
    """
    samples = data
    num_samples = len(samples)
    #print('kind of indices', samples)
    while True:
        #Generate random batch of indices
        #indices = np.random.permutation(data.count()[0])
        for batch in range(0, num_samples, batch_size):
            batch_samples = samples[batch:(batch + batch_size)]
            # Output arrays
            
            images = []
            steering_angles = []
            #x = np.empty([0, 160, 320, 3], dtype=np.float32)
            #y = np.empty([0], dtype=np.float32)

            # Read in and preprocess a batch of images
            for batch_sample in batch_samples:
                # Randomly select camera
                camera = np.random.randint(len(cameras)) if augment else 1
                print('shape af choosen camera', camera)
                # Read frame image and work out steering angle
                image = cv2.imread(os.path.join(rootPath, data[int(cameras[camera])].values[batch_sample].strip()))
                print("Shape of read image", image.shape)
                #image = cv2.imread(batch_sample[camera].strip())
                steering_angle = float(batch_sample[STEERING_ANGLE]) + left_right_steering_correction[camera]
                                
                #x = np.append(x, [image], axis=0)
                #y = np.append(y, [steering_angle])
                images.append(image)
                steering_angles.append(steering_angle)
            
            x = np.array(images)
            y = np.array(steering_angles)
            print(" Image data shape after create numpy array: {}:".format(x.shape))
            


            # # Randomly flip half of images in the batch
            # flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            # x[flip_indices] = np.fliplr(x[flip_indices])
            # #x[flip_indices] = cv2.flip(x[flip_indices],1)
            # y[flip_indices] = -y[flip_indices]
            # #x[flip_indices] = x[flip_indices, :, ::-1, :]
            # #y[flip_indices] = -y[flip_indices]
            # x_shape = x.shape
            # print("Image data shape after flipping images vertical =", x_shape)
            yield (x, y)
