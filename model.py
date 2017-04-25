import random
import csv
import cv2
import numpy as np
import os
import pandas as pd
from sklearn import model_selection
from resize_nomalize import resize_normalize
from generate_samples import generate_samples
from generator_fernando import generator_fernando
from sklearn.model_selection import train_test_split
import tensorflow as tf
#import generate_samples
#import resize_normalize
import random
# ================================================================================================================
# Read in rough balanced data Set
# ================================================================================================================
local_project_path = '../'
local_data_path = os.path.join(local_project_path, 'data')
# load balanced data set
#data_set = pd.io.parsers.read_csv(os.path.join(local_data_path, 'driving_log_balanced.csv'))
#data_set = pd.io.parsers.read_csv(os.path.join(local_data_path, 'driving_log.csv'))
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)
del(lines[0])


images =[]
measurements = []
for line in lines:
    source_path = line[0]
    filename= source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    
    measurement = float(line[3])
    measurements.append(measurement)


X_train = np.array(images)
y_train = np.array(measurements)

# delete the first row which has column names like 'left', 'steering' etc
del(lines[0])

X_train, y_valid = train_test_split(lines, test_size=0.2)
# Split data into training and validation set
#X_train, y_valid = model_selection.train_test_split(data_set, test_size=.2)

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
print('I am before call of architecture')

def architecture():
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
    model.add(Lambda(lambda x: resize_normalize(x),input_shape=(80,318,3),output_shape=(66, 200, 3)))
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
    
    model.compile(optimizer='adam', loss='mse')
    print('I am finished build the model')
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
model = architecture()
num_epochs= 15

for time in range(numTimes):
    
    #print('number of training data: ', len(X_train))
    #print('number of training data:', len(y_valid))
    print('samples_per_epoch:', X_train.shape[0])
    print('nb_val_samples:', y_valid.shape[0])
    print('number of epochs:', num_epochs)
    print('I am before call of model.fit generator')
    # training pipeline with keras
    history = model.fit_generator(#generator_fernando(X_train),
            generate_samples(X_train, local_data_path),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=num_epochs,
            validation_data=generate_samples(y_valid, local_data_path, augment=False),
            #validation_data=generator_fernando(y_valid),
            nb_val_samples=y_valid.shape[0]
            )

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
    plt.show()

    val_loss = history.history['val_loss'][0]
    if val_loss < val_best:
        val_best = val_loss

        save_model("model")

        print('Time: ', time + 1)

print ('===========================================================')
print ('traing session has finished')
print ('===========================================================')
