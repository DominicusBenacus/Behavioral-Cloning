#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following necessary files. There are more files but not every file is needed for training:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* generate_samples.py contains the augmentation steps. All camera with steering angle offset correction, flipping half of      each batch and randamise the brightness of eacht image
* resize_normalize.py to preprocess the sample data. Function is called during training using a Lambda layer
* writeup_report.md
* balancing.py

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by:

First: Start the simulator in autonomously mode.
Second:
Start the anaconda shell (in the repo directory) and activate the carnd-term1 starter kit environment
Third: Type in the shell --> python drive.py model.h5

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model consists of a convolution neural network with 3 time 5*5 and two itmes 3*3 filter sizes and depths between 24 and 64 (model.py lines 74-96)
#### 2. Attempts to reduce overfitting in the model
I added dropout on one flatten and every dense layers to prevent overfitting, and the model proved to generalise quite well. The model was trained using Adam optimiser with the default learning rate and mean squared error as a loss function. I used 20% of the training data for validation. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 99).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.I have collected a dataset containing approximately 1 hour worth of driving data around one of the given tracks. This would contain both driving in “smooth” mode (staying right in the middle of the road for the whole lap), and “recovery” mode (letting the car drive off center and then interfering to steer it back in the middle).

The second step was to augument the data set. 
* After generate the data I read in the .csv file liek described in model.py showen at start. 
##### Data augmentation

* First step I used a python script to skip all rows with a steering angle of x <=0.4 like some forum mentors described. I only skip the steering values for laps which ar recorded controled by keyboard. For all laps I recorded controled by mouse I instead used more random selection methods. In this way I did not loos so many data. Also I figured out that using mouse produces a les amount of steering absolut steering values. So with keyboard there are often values +- 8degree and higher. With mouse the range is much more smaller. However, as many pointed out, there a couple of augmentation tricks that should let you extend the dataset significantly:

###### Left and right cameras
Along with each sample we receive frames from 3 camera positions: left, center and right. Although we are only going to use central camera while driving, we can still use left and right cameras data during training after applying steering angle correction, increasing number of examples by a factor of 3.

```python
for batch_sample in batch_samples:
    #choose a random camero for read in an image
    camera = np.random.randint(len(cameras)) if augment else 1
    source_path = batch_sample[camera]
    filename = source_path.split('\\')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
```
 For each I use a steering correction depending on which camera position was choosen between 0.25 -0.3
###### Horizontal flip
For every batch we flip half of the frames horizontally and change the sign of the steering angle, thus yet increasing number of examples by a factor of 2.

```python
 # Randomly flip half of images in the batch
 flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
 x[flip_indices] = np.fliplr(x[flip_indices])
 # #x[flip_indices] = cv2.flip(x[flip_indices],1)
 y[flip_indices] = -y[flip_indices] 
``` 
 ###### Random brightness variation

 ```python
 # randomize brightness
 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
 random_brightness = .25 + np.random.uniform()
 image[:, :, 2] = image[:, :, 2] * random_brightness
 image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
```                

###### Cropping top and bottom

```python
    model.add(Cropping2D(cropping=((60,20), (1,1)), input_shape=(160,320,3)))
```
The reason is that only the road is important to make good predictions.

##### Resize and normalize

```python
    model.add(Lambda(resize_normalize, input_shape=(160, 320, 3), output_shape=(66, 200, 3)))
```
So for resize and normalize I used the Lambda layer to get the full advantage of a aws GPU.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the model shown in the introduction. Just to get running in python.
```python
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))
```

Ok second: After some research and some talks to the mentors and other students I decided to implement the nvida model with kreas in python. Later I realized that it is not everytime the best practice to use a proofen network according to the different data I recorded in this project. In my opinion the nvidea net is too big and fits more and other kind of input data better.

The CNN archtitecture like described in the model.py (# uncommended) file came out of the  [nvidia paper](https://arxiv.org/pdf/1604.07316.pdf)

Here is a model summary on a glance generated during traing session:

Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 80, 318, 3)    0           cropping2d_input_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 66, 200, 3)    0           cropping2d_1[0][0]
____________________________________________________________________________________________________
conv1 (Convolution2D)            (None, 31, 98, 24)    1824        lambda_1[0][0]
____________________________________________________________________________________________________
conv2 (Convolution2D)            (None, 14, 47, 36)    21636       conv1[0][0]
____________________________________________________________________________________________________
conv3 (Convolution2D)            (None, 5, 22, 48)     43248       conv2[0][0]
____________________________________________________________________________________________________
conv4 (Convolution2D)            (None, 3, 20, 64)     27712       conv3[0][0]
____________________________________________________________________________________________________
conv5 (Convolution2D)            (None, 1, 18, 64)     36928       conv4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           conv5[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1152)          0           flatten_1[0][0]
____________________________________________________________________________________________________
hidden1 (Dense)                  (None, 1164)          1342092     dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 1164)          0           hidden1[0][0]
____________________________________________________________________________________________________
hidden2 (Dense)                  (None, 100)           116500      dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 100)           0           hidden2[0][0]
____________________________________________________________________________________________________
hidden3 (Dense)                  (None, 50)            5050        dropout_3[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 50)            0           hidden3[0][0]
____________________________________________________________________________________________________
hidden4 (Dense)                  (None, 10)            510         dropout_4[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 10)            0           hidden4[0][0]
____________________________________________________________________________________________________
output (Dense)                   (None, 1)             11          dropout_5[0][0]
====================================================================================================
Total params: 1,595,511
Trainable params: 1,595,511
Non-trainable params: 0
____________________________________________________________________________________________________ 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I record more data in that region and record some recovery data as described inthe section of data augmentation. but it did not kept the car in the road. Because of this and the huge amount of data on this time point. I decided to reduce the complexity of the model to one I described int he next section. 

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes. 

So this is the less complex model. I decrease the amount of conv layers and also the depth of filters becuae the data we used in P3 is not so comlicated. Further more the model use max pooling and a RELU activation function. 

Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 80, 318, 3)    0           cropping2d_input_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 32, 128, 3)    0           cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 30, 126, 16)   448         lambda_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 15, 63, 16)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 13, 61, 32)    4640        maxpooling2d_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 6, 30, 32)     0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 4, 28, 64)     18496       maxpooling2d_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 2, 14, 64)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1792)          0           maxpooling2d_3[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 500)           896500      flatten_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 500)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           50100       dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 20)            2020        dropout_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             21          dense_3[0][0]
====================================================================================================
Total params: 972,225
Trainable params: 972,225
Non-trainable params: 0
____________________________________________________________________________________________________

The model includes RELU layers to introduce nonlinearity, and the data is cropped resized and normalized inside the model using a Keras cropping2D function and for resize and normalize a lambda layer (code line 74-78). I put the prprocessing inside the model architecture to achieve the complete benefit of using a GPU an the aws.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. like shown in this video

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Middle lane image][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![recovery center][image1]
![recovery left][image2]
![recovery right][image3]

So after some trainings and test sessions I often merge new traing data into th ealready used and train the model from scratch. According to time for training this was ok because I used aws. Further more I used sometimes a pretrained model I saved if I get a better loss than before. But for pretrained models one need to deal much more with hyperparameters and I want to keep things easy and get back to train the model from scratch.


Here some collection of facts I used for the final training session

##### Startpoint
number of training samples: 8289:
samples_per_epoch 8192
number of validation samples: 2073:
nb_val_epoch 2048
number of epochs: 20
##### FirstEpoch  starts with this input
Shape of image: (128, 160, 320, 3):
Shape of steering angle: (128,):
Image data shape after flipping images vertical = (128, 160, 320, 3)

##### Results after Trianing
8289/8192 [==============================] - 25s - loss: 0.0311 - val_loss: 0.0368
I saved the model
Model fit generator finished

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20. I used an adam optimizer so that manually training the learning rate wasn't necessary.

[//]: # (Image References)

[image1]: ./examples/recover/center_2017_05_01_18_43_06_626.jpg "recovery"
[image2]: ./examples/recover/left_2017_05_01_18_43_06_886.jpg "recovery"
[image3]: ./examples/recover/right_2017_05_01_18_43_06_626.jpg "recovery"
[image4]: ./examples/center_2017_05_01_18_31_43_181.jpg "Middle lane image"

