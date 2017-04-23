import csv
import cv2
import numpy as np

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)

images =[]
measurements = []
for line in lines:
    source_path = line[0]
    filename= source_path.split('/')[-1]
    current_path = '../data/IMG' + filename
    image = cv2.imread(current_path)
    images.append(image)
    
    measurement = float(line[3])
    measurements.append(measurement)


X_train = np.array(images)
y_train = np.array(measurements)

# Architectures
## Simple Model
from keras.models import Sequential,Model
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Activation, Dropout, MaxPooling2D, Flatten, Lambda, ELU
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

#trian the model
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

model.save('model.h5')

### The Nvidia architecture like described im here https://arxiv.org/pdf/1604.07316.pdf
#
#model = Sequential()
#dropout = 0.5
#nonlinear = 'tanh'
#shifting = True
#### Randomly shift up and down while preprocessing
#shift_delta = 8 if shifting else 0
#
#### Convolution layers and parameters were taken from the "nvidia paper" on end-to-end autonomous steering.
#model.add(Cropping2D(cropping=((random.uniform(60 - shift_delta , .60 + shift_delta)),(random.uniform(20 - shift_delta , .20 + shift_delta))), (0,0)), input_shape=(160,320,3)))
#model.add(Lambda(resize_normalize(image),input_shape=(160,320,3)))
#model.add(Convolution2D(24, 5, 5, name='conv1', subsample=(2, 2), activation=nonlinear))
#model.add(Convolution2D(36, 5, 5, name='conv2', subsample=(2, 2), activation=nonlinear))
#model.add(Convolution2D(48, 5, 5, name='conv3', subsample=(2, 2), activation=nonlinear))
#model.add(Convolution2D(64, 3, 3, name='conv4', activation=nonlinear))
#model.add(Convolution2D(64, 3, 3, name='conv5', activation=nonlinear))
#
#### Regression
#model.add(Flatten())
#model.add(Dropout(dropout))
#model.add(Dense(1164, name='hidden1', activation=nonlinear))
#model.add(Dropout(dropout))
#model.add(Dense(100, name='hidden2', activation=nonlinear))
#model.add(Dropout(dropout))
#model.add(Dense(50, name='hidden3', activation=nonlinear))
#model.add(Dropout(dropout))
#model.add(Dense(10, name='hidden4', activation=nonlinear))
#model.add(Dropout(dropout))
#model.add(Dense(1, name='output', activation=nonlinear))    
#   
#model.compile(optimizer='adam', loss='mse')
#
#return model

## Save Model
#
#
#from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, Callback
#from pathlib import Path
#import json
#    
#def save_model(name):
#    
#    with open(name + '.json', 'w') as output:
#        output.write(model.to_json())
#
#    model.save(name + '.h5')


#
## evaluate the trainig results
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
### plot the training and validation loss for each epoch
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model mean squared error loss')
#    plt.ylabel('mean squared error loss')
#    plt.xlabel('epoch')
#    plt.legend(['training set', 'validation set'], loc='upper right')
#    plt.show()
#    
#    val_loss = history.history['val_loss'][0]
#    if val_loss < val_best:
#        val_best = val_loss
#        
#        save_model("model")
#        
#    print('Time: ', time + 1)
