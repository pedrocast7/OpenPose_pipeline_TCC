# Author: Pedro Anderson Ferreira Castro
# Date: 24 May 2023


import numpy as np 
import math
import pandas as pd
import argparse
from Openpose_lib_functions import get_angle


parser = argparse.ArgumentParser()
parser.add_argument('--table_path', help='Path to folder containing the table file', default='C:/Users/pedro/OneDrive/Documentos/UFPA - Material/TCC STUFF/Samples_04-10/DATA_TXT_CSV/Lucas_abducao_lat_2.csv')
parser.add_argument('--table_output_name', help='Path and name of the csv file generated at the ouput', default='C:/Users/pedro/OneDrive/Documentos/UFPA - Material/TCC STUFF/Samples_04-10/DATA_TXT_CSV/OP_angle_lucas_abducao_lat_2.csv')
args = parser.parse_args()

folder_path = args.table_path
output_csv = args.table_output_name

txt_switch = 'False'

if txt_switch == 'True':
    keypoints = pd.read_table(folder_path, decimal = '.', encoding='latin-1')

    edge1_X = keypoints['T 4 X']
    edge1_Y = keypoints['T 4 Z']*(-1)

    intersec_X = keypoints['Acrômio esq. X']
    intersec_Y = keypoints['Acrômio esq. Z']*(-1)

    edge2_X = keypoints['Olécrano esq. X']
    edge2_Y = keypoints['Olécrano esq. Z']*(-1)

else:
    keypoints = pd.read_csv(folder_path)
    
    edge1_X = keypoints.neck_x
    edge1_Y = keypoints.neck_y

    intersec_X = keypoints.left_shoulder_x
    intersec_Y = keypoints.left_shoulder_y

    edge2_X = keypoints.left_elbow_x
    edge2_Y = keypoints.left_elbow_y


joint_angles = []


## get_angle(edge1(x,y),  edge2(x,y), intersection(x,y))

for index in range(len(edge1_X)):
    
    joint_angles.append(  get_angle((edge1_X[index], edge1_Y[index]),
                                     (edge2_X[index], edge2_Y[index]),
                                       (intersec_X[index], intersec_Y[index]))  )


keypoints_angles = pd.DataFrame(joint_angles)  #.transpose() one column only, no needed
keypoints_angles.columns = ['joint_angles']
keypoints_angles.to_csv(output_csv, index=False)



