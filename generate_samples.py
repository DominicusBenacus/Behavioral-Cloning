import numpy as np
import skimage.transform as sktransform
import random
import cv2
import os
from random import shuffle
import sklearn 


# set up some parameter for dealing with the different camera positions
left_right_steering_correction = [0, .25, -.25]
STEERING_ANGLE = 3


def generate_samples(samples, augment=True, batch_size=128):
    """
    Keras generator yielding batches of training/validation data.
    Applies data augmentation pipeline if `augment` is True.

    Argument pipeline contains following augmentation prcesses
    - use images from all camera. Make the data set three times bigger
    - correction of the sterring angle depending on which camera position is read in
    - flipping images verticaly
    - make random brightnes adaption  
    """
    #samples = shuffle(samples)
    num_samples = len(samples)
    batch_size = int(batch_size / 3)
    #print('kind of indices', samples)
    while True:
        # Generate random batch of indices
        #indices = np.random.permutation(data.count()[0])
        for batch in range(0, num_samples, batch_size):
            batch_samples = samples[batch:(batch + batch_size)]
            # create lists
            images = []
            steering_angles = []
            # Read in and preprocess a batch of images
            for batch_sample in batch_samples:
                for i in range(0, 3):
                    source_path = batch_sample[i]
                    filename = os.path.split(source_path)
                    #filename = source_path.split('\\')[-1]
                    current_path = os.path.join('../data/IMG/',filename)
                    #current_path = '../data/IMG/' + filename
                    image = cv2.imread(current_path)

                    # # randomize brightness
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    # random_brightness = .1 + np.random.uniform()
                    # image[:, :, 2] = image[:, :, 2] * random_brightness
                    # image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

                    images.append(image)
                    # read steering angle depending an add offset depending on
                    # the camera possition
                    steering_angle = float(batch_sample[STEERING_ANGLE])
                    #print(" Shape of steering angle ater read in: {}:".format(steering_angle))
                    steering_angle = float(steering_angle) + float(left_right_steering_correction[i])
                    #print(" Shape of steering angle after add or sub wor out: {}:".format(steering_angle))
                    steering_angles.append(steering_angle)

            #print(" Shape of steering angle after append: {}:".format(steering_angles))
            # create a numpy array because keras expect a numpy array as input
            x = np.array(images)
            y = np.array(steering_angles)
            print(" Shape of image: {}:".format(x.shape))
            print(" Shape of steering angle: {}:".format(y.shape))

            # # Randomly flip half of images in the batch
            flip_indices = random.sample(
                range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = np.fliplr(x[flip_indices])
            # #x[flip_indices] = cv2.flip(x[flip_indices],1)
            y[flip_indices] = -y[flip_indices]
            
            x_shape = x.shape
            print("Image data shape after flipping images vertical =", x_shape)
            yield sklearn.utils.shuffle(x, y)
