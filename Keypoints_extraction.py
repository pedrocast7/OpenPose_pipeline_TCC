# Author: Pedro Anderson Ferreira Castro
# Date: 24 May 2023


#./build/examples/openpose/openpose.bin --display 0 --model_pose BODY_25 
# --net_resolution -1x336 --part_candidates  --video ./examples/media/OneDrive_1_13-07-2022/video-01-rotated.mp4 
# --write_video ./examples/media/OneDrive_1_13-07-2022/video-01-rotated_output.avi 
# --write_json ./examples/media/OneDrive_1_13-07-2022/video-01-336net-jsons/



#### CODE THAT EXTRACTS THE KEYPOINTS FROM OPENPOSE JSONS



# load modules
import os
import numpy as np
import json
import pandas as pd
import argparse
from Openpose_lib_functions import get_point_estim, pixel_for_m

### Parsing paths and output .csv file name ###
parser = argparse.ArgumentParser()
parser.add_argument('--jsons_path', help='Path to folder containing the json files', default='/home/lamic/Openpose-pedro/LEMOH_EXP/arm_abduction_test/cam1_op_video_grayscale-jsons/')
parser.add_argument('--output_csv', help='Csv output file name/path', default='/home/lamic/Openpose-pedro/LEMOH_EXP/cam1_op_video_grayscale_new.csv')
args = parser.parse_args()

# path to .json files
path_to_data_files = args.jsons_path
#path_to_data_files = 'C:/Users/Clebson/Downloads/lemoh012'

#name for the csv that will be created
file_name_csv = args.output_csv

## vectors that will store the keypoints of all frames
lkx = []
lky = []
rhx = []
rhy = []
rkx = []
rky = []
rax = []
ray = []
ra = []
rk = []
rsx = []
rsy = []
L5_pair = []
rs_list = []
nkx = []
nky = []
mhx = []
mhy = []
lhx = []
lhy = []
lax = []
lay = []

### upper body
lshx = []
lshy = []
lelx = []
lely = []
lwrsx = []
lwrsy = []


## New point estimated (Center of Mass)
nk_pair = []
md_hip_pair = []

L5_pair =[]
L5_x = []
L5_y = []
L5_list = []






# read the json files and stores all x and y points of interest
json_files = [pos_json for pos_json in sorted(os.listdir(
    path_to_data_files)) if pos_json.endswith('.json')]

for index, js in enumerate(json_files):
    f = open(os.path.join(path_to_data_files, js), 'r')
    data = f.read()
    jsondata = json.loads(data)
    
    sorted_keypoints = []
    counter = 0
    keypoints_2d_full = jsondata["people"][0]["pose_keypoints_2d"]
    

    for i in range(0, len(keypoints_2d_full)-3, 3):
        counter = counter+1
        sorted_keypoints.insert(counter, keypoints_2d_full[i:i+3])


    neck = sorted_keypoints[1][:2]

    right_shoulder = sorted_keypoints[2][:2]
    right_elbow = sorted_keypoints[3][:2]
    right_wrist = sorted_keypoints[4][:2]

    left_shoulder = sorted_keypoints[5][:2]
    left_elbow = sorted_keypoints[6][:2]
    left_wrist = sorted_keypoints[7][:2]

    mid_hip = sorted_keypoints[8][:2]

    right_hip = sorted_keypoints[9][:2]
    right_knee = sorted_keypoints[10][:2]
    right_ankle = sorted_keypoints[11][:2]

    left_hip = sorted_keypoints[12][:2]
    left_knee = sorted_keypoints[13][:2]
    left_ankle = sorted_keypoints[14][:2]


   

    rhy.insert(index, right_hip[1])
    rhx.insert(index, right_hip[0])
    rky.insert(index, right_knee[1])
    rkx.insert(index, right_knee[0])
    rk.insert(index, right_knee)
    ray.insert(index, right_ankle[1])
    rax.insert(index, right_ankle[0])
    lky.insert(index, left_knee[1])
    lkx.insert(index, left_knee[0])
    nkx.insert(index, neck[0])
    nky.insert(index, neck[1])
    mhx.insert(index, mid_hip[0])
    mhy.insert(index, mid_hip[1])
    lhx.insert(index, left_hip[0])
    lhy.insert(index, left_hip[1])
    lax.insert(index, left_ankle[0])
    lay.insert(index, left_ankle[1])

    lshx.insert(index, left_shoulder[0])
    lshy.insert(index, left_shoulder[1])

    lelx.insert(index, left_elbow[0])
    lely.insert(index, left_elbow[1])

    lwrsx.insert(index, left_wrist[0])
    lwrsy.insert(index, left_wrist[1])

    #neck = np.array([nkx, nky])
    #nk_pair.insert(index, neck)
    #mid_hip = np.array([mhx, mhy])
    #md_hip_pair.insert(index, mid_hip)


##### Work on logic to estimate point
#print(len(neck[:][0]))
for item in range(index+1):
    pair_4_estm = get_point_estim(nkx[item], nky[item], mhx[item], mhy[item])
    L5_pair.insert(item, pair_4_estm)

for point in range(len(L5_pair)):
        x_axis = L5_pair[point][0]
        y_axis = L5_pair[point][1]
        L5_x.insert(point, x_axis)
        L5_y.insert(point, y_axis)




# conversion pixel for cm
# Saida(cm) = 0.2* Entrada(pixel)
print(range(len(lax)))
def rem_nan(vector):
    #removes nan data from a given array 
    
    #pos_nan = np.where(pd.isnull(vector))
    
    new_vec = np.trim_zeros(vector)#list(np.delete(vector, pos_nan))


    return new_vec

    # nan_array = pd.isnull(vector)
    # for f_value in nan_array:
        # 
        # if (nan_array[f_value]).any() :   ## when trues enter the condition            
            # vector = np.delete(vector, f_value)

    # cleaned_vect = []
# 
    # for f_value in vector:
        # if f_value != None:
            # cleaned_vect.append(f_value)
# 
    # return cleaned_vect
    

# for element in myList:
#     if not math.isnan(element):
#         newList.append(element)


rhx = rem_nan(rhx)
rhy = rem_nan(rhy)
rkx = rem_nan(rkx)
rky = rem_nan(rky)
rax = rem_nan(rax)
ray = rem_nan(ray)
rsx = rem_nan(rsx)
rsy = rem_nan(rsy)
lkx = rem_nan(lkx)
lky = rem_nan(lky)
nkx = rem_nan(nkx)
nky = rem_nan(nky)
mhx = rem_nan(mhx)
mhy = rem_nan(mhy)
lhx = rem_nan(lhx)
lhy = rem_nan(lhy)
lax = rem_nan(lax)
lay = rem_nan(lay)


lshx = rem_nan(lshx)
lshy = rem_nan(lshy)
lelx = rem_nan(lelx)
lely = rem_nan(lely)
lwrsx = rem_nan(lwrsx)
lwrsy = rem_nan(lwrsy)
L5_x = rem_nan(L5_x)
L5_y = rem_nan(L5_y)




# rhx = smooth_savgol(rhx,15,5,'interp')
# rhy = smooth_savgol(rhy,15,5,'interp')
# rkx = smooth_savgol(rkx,15,5,'interp')
# rky = smooth_savgol(rky,15,5,'interp')
# rax = smooth_savgol(rax,15,5,'interp')
# ray = smooth_savgol(ray,15,5,'interp')
#rsx = smooth_savgols(rsx,15,5,'interp')
#rsy = smooth_savgols(rsy,15,5,'interp')
# lkx = smooth_savgol(lkx,15,5,'interp')
# lky = smooth_savgol(lky,15,5,'interp')
# nkx = smooth_savgol(nkx,15,5,'interp')
# nky = smooth_savgol(nky,15,5,'interp')
# mhx = smooth_savgol(mhx,15,5,'interp')
# mhy = smooth_savgol(mhy,15,5,'interp')
# lhx = smooth_savgol(lhx,15,5, 'interp')
# lhy = smooth_savgol(lhy,15,5, 'interp')
# lax = smooth_savgol(lax,15,5, 'interp')
# lay = smooth_savgol(lay,15,5, 'interp')
# 
# lshx = smooth_savgol(lshx, 15, 5, 'interp')
# lshy = smooth_savgol(lshy, 15, 5, 'interp')
# lelx = smooth_savgol(lelx, 15, 5, 'interp')
# lely = smooth_savgol(lely, 15, 5, 'interp')
# lwrsx = smooth_savgol(lwrsx, 15, 5, 'interp')
# lwrsy = smooth_savgol(lwrsy, 15, 5, 'interp')


# rhx = pop_outlayers_interpol(rhx,46,15)
# rhy = pop_outlayers_interpol(rhy,46,15)
# rkx = pop_outlayers_interpol(rkx,46,15)
# rky = pop_outlayers_interpol(rky,46,15)
# rax = pop_outlayers_interpol(rax,46,15)
# ray = pop_outlayers_interpol(ray,46,15)
# rsx =pop_outlayers_interpol(rsx,46,15)
# rsy =pop_outlayers_interpol(rsy,46,15)
# lkx = pop_outlayers_interpol(lkx,46,15)
# lky = pop_outlayers_interpol(lky,46,15)



#print('array em pixel: ', rhx)

#rhx = pixel_for_m(rhx, 0.157892576)     #0.00195 Old value
#rhy = pixel_for_m(rhy, 0.157892576)
#rkx = pixel_for_m(rkx, 0.157892576)
#rky = pixel_for_m(rky, 0.157892576)
#rax = pixel_for_m(rax, 0.157892576)
#ray = pixel_for_m(ray, 0.157892576)
#rsx = pixel_for_m(rsx, 0.157892576)
#rsy = pixel_for_m(rsy, 0.157892576)
#lkx = pixel_for_m(lkx, 0.157892576)
#lky = pixel_for_m(lky, 0.157892576)
# nkx = pixel_for_m(nkx, 0.157892576)
# nky = pixel_for_m(nky, 0.157892576)
# mhx = pixel_for_m(mhx, 0.157892576)
# mhy = pixel_for_m(mhy, 0.157892576)
#lhx = pixel_for_m(lhx, 0.157892576)
#lhy = pixel_for_m(lhy, 0.157892576)
#lax = pixel_for_m(lax, 0.157892576)
#lay = pixel_for_m(lay, 0.157892576)

lshx = pixel_for_m(lshx, 0.116083011)
lshy = pixel_for_m(lshy, 0.116083011)
lelx = pixel_for_m(lelx, 0.116083011)
lely = pixel_for_m(lely, 0.116083011)
lwrsx = pixel_for_m(lwrsx, 0.116083011)
lwrsy = pixel_for_m(lwrsy, 0.116083011)

# L5_x = pixel_for_m(L5_x, 0.116083011)
# L5_y = pixel_for_m(L5_y, 0.116083011)

#print('array em metros: ', rhx)


# dataframe and csv file
anatomicals = [rhx, rhy, rkx, rky, rsx, rsy, rax, ray, lkx,
               lky, nkx, nky, mhx, mhy, lhx, lhy, lax, lay,
               lshx, lshy, lelx, lely, lwrsx, lwrsy,
               L5_x, L5_y
               ]
anatomical_points = pd.DataFrame(anatomicals).transpose()
anatomical_points.columns = ['right_hip_x', 'right_hip_y', 'right_knee_x', 'right_knee_y', 'right_shank_x',
'right_shank_y', 'right_ankle_x', 'right_ankle_y', 'left_knee_x', 'left_knee_y', 'neck_x', 'neck_y', 'mid_hip_x',
'mid_hip_y', 'left_hip_x', 'left_hip_y', 'left_ankle_x', 'left_ankle_y',
'left_shoulder_x', 'left_shoulder_y', 'left_elbow_x', 'left_elbow_y', 'left_wrist_x', 'left_wrist_y',
'L5_x', 'L5_y'
                            ]
anatomical_points.to_csv(file_name_csv, index=False)


