import csv
import cv2
import numpy as np
lines = []
with open('../data/driving_log.csv') as csvfil:
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



# crop the images to 64*64



# normalize and meancenter the images


# Architectures
## Simple Model
from keras.models import Sequential,Model
from keras.layers import Flatten, Dense,Lambda
from keras.layers import Convolution2D
from keras.pooling import Convolution2D


model = Sequential
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

#trian the model
model.fit(X_train,y_train,validation_split=0.2,shufffle=TRUE,nb_epoch=7)

model.save('model.h5')

## The Nvidia architecture like described im here https://arxiv.org/pdf/1604.07316.pdf

#model = Sequential()
#dropout = 0.5
#nonlinear = 'tanh'
#
#### Convolution layers and parameters were taken from the "nvidia paper" on end-to-end autonomous steering.
#model.add(Lambda(normalize, input_shape=(160, 320, 3), output_shape=(66, 200, 3)))
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


# train the model


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