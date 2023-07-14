# Author: Pedro Anderson Ferreira Castro
# Date: 24 May 2023


import numpy as np 
import math
import pandas as pd
import argparse
from Openpose_lib_functions import get_angle


parser = argparse.ArgumentParser()
parser.add_argument('--table_path', help='Path to folder containing the table file', default='/home/lamic/Openpose-pedro/scripts_openpose/codigos/video-01-336net.csv')
parser.add_argument('--table_output_name', help='Path and name of the csv file generated at the ouput', default='/home/lamic/Openpose-pedro/scripts_openpose/codigos/test_angle.csv')
args = parser.parse_args()

folder_path = args.table_path
output_csv = args.table_output_name


keypoints = pd.read_csv(folder_path)

hip_points_X = keypoints.right_hip_x
hip_points_Y = keypoints.right_hip_y

knee_points_X = keypoints.right_knee_x
knee_points_Y = keypoints.right_knee_y

ankle_points_X = keypoints.right_ankle_x
ankle_points_Y = keypoints.right_ankle_y

joint_angles = []


for index in range(len(hip_points_X)):
    
    joint_angles.append(  get_angle((hip_points_X[index], hip_points_Y[index]), (ankle_points_X[index], ankle_points_Y[index]), (knee_points_X[index], knee_points_Y[index]))  )


keypoints_angles = pd.DataFrame(joint_angles)  #.transpose() one column only, no needed
keypoints_angles.columns = ['joint_angles']
keypoints_angles.to_csv(output_csv, index=False)



