import csv
import cv2
import numpy as np
import os
import pandas as pd

print('start of balancing')
#                                                cam,    cam,  cam 
# read in the recoded data set froma csv fromat [center, left, right, steering, throttle, brake, speed]
samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
      samples.append(sample)
del(samples[0])

print(" shape of the first row of samples after imread: {}:".format(samples[0]))

for line in range(len(samples)):
    

#balance data and save it to a neew csv fil


print('end of balancing')    
balanced.to_csv('data/driving_log_balanced.csv', index=False)
print('balanced data set saved to data/driving_log_balanced.csv')    
print('========================================================')
print('balanceing process done')
