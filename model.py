import random
import csv
import cv2
import numpy as np
import os
import pandas as pd
from sklearn import model_selection
#from resize_nomalize import resize_normalize
from generate_samples import generate_samples
from generator_fernando import generator_fernando
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
import gc
from keras import backend as K
# ================================================================================================================
# Read in rough balanced data Set
# ================================================================================================================
samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
      samples.append(sample)
del(samples[0])

print(" shape of the first row of samples after imread: {}:".format(samples[0]))

# Split data into training and validation set
#sklearn.model_selection.StratifiedShuffleSplit
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(" shape of the training_samples: {}:".format(train_samples[0]))
print(" shape of the validation_samples: {}:".format(validation_samples[0]))

# ================================================================================================================
# Model Architectures
# The Nvidia architecture like described im here https://arxiv.org/pdf/1604.07316.pdf
# ================================================================================================================
from keras.models import Sequential,Model
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Activation, Dropout, MaxPooling2D, Flatten, Lambda, ELU
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam
from keras import models, optimizers, backend
from keras.models import load_model

print('I am before call of architecture')

def architecture():
    def resize_normalize(image):
        import cv2
        from keras.backend import tf as ktf    
        """
        Applies preprocessing pipeline to an image: crops `top` and `bottom`
        portions of image, resizes to 66*200 px and scales pixel values to [0, 1].
        """
        # resize
        #image = cv2.resize(image, (66, 200)) #first try
        resized = ktf.image.resize_images(image, (66, 200))
        #normalize
        resized = resized/255.0 - 0.5

        return resized

    print('I am inside call of architecture')
    #initialize model
    model = Sequential()
    dropout = 0.5
    nonlinear = 'tanh'
    #shifting = True
    ### Randomly shift up and down while preprocessing
    #shift_delta = 8 if shifting else 0
    print('I am before call of cropping layer')
    ### Convolution layers and parameters were taken from the "nvidia paper" on end-to-end autonomous steering.
    #model.add(Cropping2D(cropping=(((random.uniform(60 - shift_delta , 60 + shift_delta)),(random.uniform(20 - shift_delta , 20 + shift_delta))), (1,1)), input_shape=(160,320,3)))

    model.add(Cropping2D(cropping=((60,20), (1,1)), input_shape=(160,320,3)))

    print('I am before call of Lambda')
    model.add(Lambda(resize_normalize, input_shape=(160, 320, 3), output_shape=(66, 200, 3)))
    #model.add(Lambda(lambda x: resize_normalize(x), input_shape=(80,318,3), output_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, name='conv1', subsample=(2, 2), activation=nonlinear))
    model.add(Convolution2D(36, 5, 5, name='conv2', subsample=(2, 2), activation=nonlinear))
    model.add(Convolution2D(48, 5, 5, name='conv3', subsample=(2, 2), activation=nonlinear))
    model.add(Convolution2D(64, 3, 3, name='conv4', activation=nonlinear))
    model.add(Convolution2D(64, 3, 3, name='conv5', activation=nonlinear))

    ### Regression
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1164, name='hidden1', activation=nonlinear))
    model.add(Dropout(dropout))
    model.add(Dense(100, name='hidden2', activation=nonlinear))
    model.add(Dropout(dropout))
    model.add(Dense(50, name='hidden3', activation=nonlinear))
    model.add(Dropout(dropout))
    model.add(Dense(10, name='hidden4', activation=nonlinear))
    model.add(Dropout(dropout))
    model.add(Dense(1, name='output', activation=nonlinear))    
    
    #model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')
    model.compile(optimizer='adam', loss='mse')
    print('I am finished build the model')
    print(model.summary())
    return model

# Save Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, Callback
from pathlib import Path
import json

def save_model(name):
    
    with open(name + '.json', 'w') as output:
        output.write(model.to_json())

    model.save(name + '.h5')
    print('I saved the model')
# ================================================================================================================
# Training
# ================================================================================================================
numTimes = 1
val_best = 999
# preload e weights if u train the model after append more traing data makes training faster.
model = load_model('model.h5')
print(model.summary())

#model = architecture()
num_epochs= 10
batch_size = 128


#define the input data for the model.fit.generator
print(" number of training samples: {}:".format(len(train_samples)))
samples_per_epoch = len(train_samples) - (len(train_samples) % batch_size)
print('samples_per_epoch',samples_per_epoch)
print(" number of validation samples: {}:".format(len(validation_samples)))
nb_val_samples=len(validation_samples) - (len(validation_samples) % batch_size)
print('nb_val_epoch',nb_val_samples)
print('number of epochs:', num_epochs)
print('I am before call of model.fit generator')
# training pipeline with keras
history = model.fit_generator(#generator_fernando(X_train),
        generate_samples(train_samples),
        samples_per_epoch=samples_per_epoch,
        nb_epoch=num_epochs,
        validation_data=generate_samples(validation_samples, augment=False),
        #validation_data=generator_fernando(y_valid),
        nb_val_samples=nb_val_samples
        )

#conditioned save mdoel routine
val_loss = history.history['val_loss'][0]
if val_loss < val_best:
    val_best = val_loss
    save_model("model")

print('Model fit generator finished')
print(history.history.keys())
# ================================================================================================================
# Evaluation of the trainig results
# ================================================================================================================
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## plot the training and validation loss for each epoch
print('I am ready to plot the evaluation')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show(block=True)

gc.collect()
K.clear_session()


print('===========================================================')
print('traing session has finished')
print('===========================================================')
