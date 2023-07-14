# Author: Pedro Anderson Ferreira Castro
# Date: 24 May 2023

# importing the module 
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats
from Openpose_lib_functions import time_vec, displac_axis, get_norm_vec, variance_over_samples

#### Videos are in the same resolution (1080x1920)

lemoh_csv_path = '/home/lamic/Openpose-pedro/LEMOH_EXP/variance_analysis_lemoh_new.csv'
laps_csv_path = '/home/lamic/Openpose-pedro/LEMOH_EXP/variance_analysis_laps_new.csv'

lemoh_data = pd.read_csv(lemoh_csv_path)
laps_data = pd.read_csv(laps_csv_path)

### loading neck reference point
lemoh_neck_x = lemoh_data['neck_x']
lemoh_neck_y = lemoh_data['neck_y']

laps_neck_x = laps_data['neck_x']
laps_neck_y = laps_data['neck_y']


### loading mid hip point
lemoh_mh_x = lemoh_data['mid_hip_x']
lemoh_mh_y = lemoh_data['mid_hip_y']

laps_mh_x = laps_data['mid_hip_x']
laps_mh_y = laps_data['mid_hip_y']

### L5 Point
lemoh_L5_x = lemoh_data['L5_x']
lemoh_L5_y = lemoh_data['L5_y']

laps_L5_x = laps_data['L5_x']
laps_L5_y = laps_data['L5_y']


### Time vector for plots
frames_num1 = len(lemoh_mh_x)
frames_num2 = len(laps_mh_x)
fps1 = 60
fps2 = 30

time_vec_lemoh = time_vec('/home/lamic/Openpose-pedro/LEMOH_EXP/variance_analysis_lemoh.avi', 'lemoh')
time_vec_laps = time_vec('/home/lamic/Openpose-pedro/LEMOH_EXP/variance_analysis_laps.mp4', 'laps')


##### Normalizing data
## neck - 0 // mid hip - 1


lemoh_cap = displac_axis(lemoh_neck_x, lemoh_neck_y, lemoh_mh_x, lemoh_mh_y, lemoh_L5_x, lemoh_L5_y, 3400)

laps_cap = displac_axis(laps_neck_x, laps_neck_y, laps_mh_x, laps_mh_y, laps_L5_x, laps_L5_y, 34)



### Point displacement over seconds
# figure, (ax1, ax2) = plt.subplots(2)
# 
# Lemoh
# ax1.plot(time_vec_lemoh, lemoh_cap, 'r', label='LEMOH')
# ax1.set_title("Lemoh axis L5 distance displacement")
# ax1.set(xlabel='Time(s)', ylabel='Displacement(normalized)')
# ax1.grid('True')
# ax1.legend()
# 
# Laps
# ax2.plot(time_vec_laps, laps_cap, 'b', label='LaPS')
# ax2.set_title("Laps L5 distance displacement")
# ax2.set(xlabel='Time(s)', ylabel='Displacement(normalized)')
# ax2.grid('True')
# ax2.legend()
# 
# plt.show()



######## L5 Point plot #####################
figure, (ax1, ax2) = plt.subplots(2)

## Lemoh
ax1.plot(time_vec_lemoh, lemoh_L5_y, 'r', label='LEMOH')
ax1.set_title("Lemoh Y raw L5 displacement")
ax1.set(xlabel='Time(s)', ylabel='Displacement')
ax1.grid('True')
ax1.legend()

## Laps
ax2.plot(time_vec_laps, laps_L5_y, 'b', label='LaPS')
ax2.set_title("Laps Y raw L5 displacement")
ax2.set(xlabel='Time(s)', ylabel='Displacement')
ax2.grid('True')
ax2.legend()

plt.show()


norm_vec_lemoh = get_norm_vec(lemoh_neck_x[3400], lemoh_neck_y[3400], lemoh_mh_x[3400], lemoh_mh_y[3400])
norm_vec_laps = get_norm_vec(laps_neck_x[34], laps_neck_y[34], laps_mh_x[34], laps_mh_y[34])



### Calculating windowed variance of Lemoh and Laps data



#lemoh_va_dist = variance_over_samples(lemoh_cap, 50, 50)
#lemoh_va_dist_y = variance_over_samples(lemoh_cap, 50, 50)

#laps_va_dist = variance_over_samples(laps_cap, 50, 50)
#laps_va_dist_y = variance_over_samples(laps_cap, 50, 50)

#lemoh_va_L5_x = variance_over_samples(lemoh_L5_x, 100, 100)
lemoh_va_L5_y, lemoh_std_dev_L5_y = variance_over_samples(lemoh_L5_y, 100, 100, norm_vec_lemoh)

#laps_va_L5_x = variance_over_samples(laps_L5_x, 10, 10)
laps_va_L5_y, laps_std_dev_L5_y = variance_over_samples(laps_L5_y, 25, 25, norm_vec_laps)


variance_full_norm_lemoh = (stats.variance(lemoh_L5_y))/norm_vec_lemoh
variance_full_norm_laps = (stats.variance(laps_L5_y))/norm_vec_laps

print('LEMOH VARIANCE FULL (NORMALIZED) {}'.format(variance_full_norm_lemoh))
print('LAPS VARIANCE FULL (NORMALIZED) {}'.format(variance_full_norm_laps))

### Variance of distance vector over seconds
#figure, (ax1, ax2) = plt.subplots(2)

# ## Lemoh
# ax1.plot(lemoh_va_dist, 'r', label='LEMOH')
# ax1.set_title("Lemoh Y axis L5 variance of distance")
# ax1.set(xlabel='Windows', ylabel='Variance')
# ax1.grid('True')
# ax1.legend()


# ## Laps
# ax2.plot(laps_va_dist, 'b', label='LaPS')
# ax2.set_title("Laps Y axis L5 variance of distance")
# ax2.set(xlabel='Windows', ylabel='Variance')
# ax2.grid('True')
# ax2.legend()

# plt.show()


### Variance of point displacement over seconds
figure, (ax1, ax2) = plt.subplots(2)

## Lemoh
ax1.plot(lemoh_va_L5_y, 'r', label='LEMOH')
ax1.set_title("Lemoh Y axis L5 variance of point displacement")
ax1.set(xlabel='Windows', ylabel='Variance')
ax1.grid('True')
ax1.legend()


## Laps
ax2.plot(laps_va_L5_y, 'b', label='LaPS')
ax2.set_title("Laps Y axis L5 variance of point displacement")
ax2.set(xlabel='Windows', ylabel='Variance')
ax2.grid('True')
ax2.legend()

plt.show()

### Standard deviation of point displacement over seconds
figure, (ax1, ax2) = plt.subplots(2)

## Lemoh
ax1.plot(lemoh_std_dev_L5_y, 'r', label='LEMOH')
ax1.set_title("Lemoh Y axis L5 standard deviation of point displacement")
ax1.set(xlabel='Windows', ylabel='Std deviation')
ax1.grid('True')
ax1.legend()


## Laps
ax2.plot(laps_std_dev_L5_y, 'b', label='LaPS')
ax2.set_title("Laps Y axis L5 standard deviation of point displacement")
ax2.set(xlabel='Windows', ylabel='Std deviation')
ax2.grid('True')
ax2.legend()

plt.show()
