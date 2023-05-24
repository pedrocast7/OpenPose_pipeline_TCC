# importing the module 
import cv2 
import numpy as np
from Openpose_lib_functions import convert_video2grayscale


video_path = '/home/lamic/Openpose-pedro/LEMOH_EXP/arm_abduction_test/cam1_op_video.mp4'
output_path = '/home/lamic/Openpose-pedro/LEMOH_EXP/arm_abduction_test/op_video_presentation_grayscale.mp4'


convert_video2grayscale(video_path, output_path, 58)

