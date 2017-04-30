import csv
import os
print('start of balancing')
#                                                cam,    cam,  cam 
# read in the recoded data set froma csv fromat [center, left, right, steering, throttle, brake, speed]
with open("driving_log.csv", "rb") as infile, open("driving_log_balanced.csv", "wb") as outfile:
    reader = csv.reader(infile)
    next(reader, None)  # skip the headers
    writer = csv.writer(outfile)
    for row in reader:
        steering_angle = abs(float(row[3]))
        print('steering value = ',steering_angle)
        if steering_angle >= 0.05:
            writer.writerow(row)

#balance data and save it to a neew csv fil
print('balanced data set saved to data/driving_log_balanced.csv')    
print('========================================================')
print('balanceing process done')
