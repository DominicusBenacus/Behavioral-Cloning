import csv
import cv2
import numpy as np
import os
import pandas as pd


print('start of balancing')
local_project_path = '/'
local_data_path = os.path.join(local_project_path, 'data')
# Read the data
data_set = pd.io.parsers.read_csv(os.path.join(local_data_path, 'driving_log.csv'))
#balance data and save it to a neew csv fil
balanced = pd.DataFrame() 	# Balanced dataset
bins = 1000 				# N of bins
bin_n = 200 				# N of examples to include in each bin (at most)

start = 0
print('start of for loop')
for end in np.linspace(0, 1, num=bins):  
    data_set_range = data_set[(np.absolute(data_set.steering) >= start) & (np.absolute(data_set.steering) < end)]
    range_n = min(bin_n, data_set_range.shape[0])
    balanced = pd.concat([balanced, data_set_range.sample(range_n)])
    start = end
print('end of balancing')    
balanced.to_csv('data/driving_log_balanced.csv', index=False)
print('balanced data set saved to data/driving_log_balanced.csv')    
print('========================================================')
print('balanceing process done')