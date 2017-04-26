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


def generate_samples(data, root_path, augment=True):
    """
    Keras generator yielding batches of training/validation data.
    Applies data augmentation pipeline if `augment` is True.

    Argument pipeline contains following augmentation prcesses
    - use images from all camera. Make the data set three times bigger
    -  
    """
    samples = data
    num_samples = len(data)
    #print('kind of indices', samples)
    batch_size = 128
    while True:
        #Generate random batch of indices
        #indices = np.random.permutation(data.count()[0])
        for batch in range(0, num_samples, batch_size):
            batch_samples = samples[batch:(batch + batch_size)]
            # Output arrays
            
            x = np.empty([0, 160, 320, 3], dtype=np.float32)
            x_shape = x.shape
            print("image shape after creation of np.empty array = ", x_shape)
            y = np.empty([0], dtype=np.float32)
            # Read in and preprocess a batch of images
            for batch_sample in batch_samples:
            #     x_shape = x.shape
            #     print("image shape direct after enter the for loop = ", x_shape)
            #     # Randomly select camera
                camera = np.random.randint(len(cameras)) if augment else 1
                print('shape af choosen camera', camera)
            #     x_shape = x.shape
            #     print("image shape after random selection of camera = ", x_shape)
            #     # Read frame image and work out steering angle
            #     #image = cv2.imread(os.path.join(root_path, data[cameras[camera]].values[i].strip()))
                image = cv2.imread(batch_sample[camera].strip())
                print('we are now after imread')
            #     #image = cv2.imread(batch_sample[cameras[camera]].strip())
            #     x_shape = x.shape
            #     print("image shape after cv2.imread = ", x_shape)
            #     steering_angle = samples.steering.values[batch_sample] + left_right_steering_correction[camera]
                steering_angle = float(batch_sample['steering_angle']) + left_right_steering_correction[camera]
            #   steering_angle = batch_sample['steering'] + left_right_steering_correction[camera] #Subodth

                print('we are now after reading steering_angle')


            #     steering_angle = float(batch_sample[3]) + left_right_steering_correction[camera]
            #     x_shape = x.shape
            #     print("image read of steering angle = ", x_shape)
	        #     # Append to batch
                x = np.append(x, [image], axis=0)
                y = np.append(y, [steering_angle])
            # x_shape = x.shape
            # print("Image data shape ater np.append(x,[image],axis=0)=", x_shape)

            # Randomly flip half of images in the batch
            flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = np.fliplr(x[flip_indices])
            #x[flip_indices] = cv2.flip(x[flip_indices],1)
            y[flip_indices] = -y[flip_indices]
            x_shape = x.shape
            print("Image data shape after flipping images vertical =", x_shape)
            yield (x, y)
