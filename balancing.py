import csv
import cv2
import numpy as np
import os
import pandas as pd

print('start of balancing')
#                                                cam,    cam,  cam 
# read in the recoded data set froma csv fromat [center, left, right, steering, throttle, brake, speed]
with open("../data/driving_log.csv", "rb") as infile, open("../data/driving_log_balanced.csv", "wb") as outfile:
   reader = csv.reader(infile)
   next(reader, None)  # skip the headers
   writer = csv.writer(outfile)
   for row in reader:       
        measurement = float(row[3])
       # process each row
       writer.writerow(row)

print(" shape of the first row of samples after imread: {}:".format(samples[0]))

for line in range(len(samples)):
    

#balance data and save it to a neew csv fil


print('end of balancing')    
balanced.to_csv('data/driving_log_balanced.csv', index=False)
print('balanced data set saved to data/driving_log_balanced.csv')    
print('========================================================')
print('balanceing process done')
