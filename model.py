import csv
import cv2
import numpy as np
import os
import pandas as pd

# ================================================================================================================
# Read in rough balanced data Set
# ================================================================================================================
local_project_path = '/'
# load balanced data set
#data_set = pd.io.parsers.read_csv(os.path.join(local_data_path, 'driving_log_balanced.csv'))
data_set = pd.io.parsers.read_csv(os.path.join(local_data_path, 'driving_log.csv'))
# Split data into training and validation set
X_train, y_valid = model_selection.train_test_split(data_set, test_size=.2)

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

#initialize model
model = Sequential()
dropout = 0.5
nonlinear = 'tanh'
shifting = True
### Randomly shift up and down while preprocessing
shift_delta = 8 if shifting else 0

### Convolution layers and parameters were taken from the "nvidia paper" on end-to-end autonomous steering.
model.add(Cropping2D(cropping=((random.uniform(60 - shift_delta , 60 + shift_delta)),(random.uniform(20 - shift_delta , 20 + shift_delta))), (0,0)), input_shape=(160,320,3))
model.add(Lambda(resize_normalize(image),input_shape=(160,320,3)))
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

return model

# ================================================================================================================
# Evaluation of the trainig results
# ================================================================================================================
history = model.fit_generator(
        generate_samples(X_train, local_data_path),
        samples_per_epoch=X_train.shape[0],
        nb_epoch=15,
        validation_data=generate_samples(X_valid, local_data_path, augment=False),
        nb_val_samples=y_valid.shape[0]
        )

# Save Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, Callback
from pathlib import Path
import json

def save_model(name):
    
    with open(name + '.json', 'w') as output:
        output.write(model.to_json())

    model.save(name + '.h5')

# ================================================================================================================
# Evaluation of the trainig results
# ================================================================================================================
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
## plot the training and validation loss for each epoch
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
