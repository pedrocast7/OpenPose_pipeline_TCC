### Code that stores all the functions used for each individual analysis: Convert_video2grayscale,
# Keypoints_Analysis_generic, Variance_openpose_analysis, Keypoints_extraction, Joint angle and angle_analysis

# Author: Pedro Anderson Ferreira Castro
# Date: 24 May 2023

import matplotlib.pyplot as plt # graphs
import numpy as np # math functions
from scipy import stats # stats functions
import statistics as stats
import math # math specific functions
from mpl_point_clicker import clicker
from scipy.ndimage.interpolation import shift
import cv2 # to deal with images too
import pywt
import scipy.interpolate as interp
from scipy.signal import savgol_filter


def convert_video2grayscale(video_path:str, output_path:str, fps:int):
    ### Takes a given video and converts to grayscale, saving them in the output_path
    
    # reading the video 
    source = cv2.VideoCapture(video_path) 

    # We need to set resolutions. 
    # so, convert them from float to integer. 
    frame_width = int(source.get(3)) 
    frame_height = int(source.get(4)) 
    
    size = (frame_width, frame_height) 

    result = cv2.VideoWriter(output_path,  
                cv2.VideoWriter_fourcc(*'MJPG'), 
                fps, size, 0) 
    
    # running the loop 
    while True: 
    
        # extracting the frames 
        ret, img = source.read() 
        if ret == False:
            break

        # converting to gray-scale 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        # write to gray-scale 
        result.write(gray)

        # displaying the video
        resized = cv2.resize(gray, (540,960))
 
        cv2.imshow("Live", resized) 
    
        # exiting the loop 
        key = cv2.waitKey(1) 
        if key == ord("q"): 
            break
        
    # closing the window 
    cv2.destroyAllWindows() 
    source.release()


def scale_and_offset(input_signal: np.array, normalize_i_o:str):
  ### corrects the scale and the offset of the function


    output_sig = input_signal - np.mean(input_signal) # offset
    
    if normalize_i_o == 'y':
        output =  output_sig/np.max(np.abs(output_sig)) # normalizes amplitude
    else:
        output = output_sig  ## without normalization
    
    return output

def align_signals(a:np.array, b:np.array):
  ### takes the signal and align it base on lag value of Cross-correlation
  xcorr = plt.xcorr(a, b, maxlags=80, normed=True, lw=2)
  
  lags = xcorr[0]
  c_values = xcorr[1]
  print('Max Xcorr lag value found is {}'.format(lags[np.argmax(c_values)]))
  aligned_vec = shift(b, lags[np.argmax(c_values)], cval=np.nan)
  return aligned_vec


def select_signals_area(lemoh, openpose):

    fig, axs = plt.subplots(2)
    axs[0].plot(lemoh,'r')
    axs[0].grid()
    axs[1].plot(openpose,'b')
    axs[1].grid()
    axs[0].set_title("LEMOH Data X")
    axs[1].set_title("OpenPose Data X")
    fig.suptitle('LEMOH & Openpose')
    
    klicker_lemoh = clicker(axs[0],["lemoh"], markers=["x"])
    klicker_openpose = clicker(axs[1],["openpose"], markers=["o"])
    plt.show()
    
    lhm_trim =  list(((klicker_lemoh.get_positions()).values()))
    op_trim = list((klicker_openpose.get_positions()).values())
    
    lhm_trim = lhm_trim[0]
    op_trim = op_trim[0]
    
    lhm_trim = np.array_split(lhm_trim,4)
    op_trim = np.array_split(op_trim,4)
    
    lhminf = np.ndarray.tolist(lhm_trim[0])
    lhminf = np.array_split(lhminf[0],2)
    lhmsup = np.ndarray.tolist(lhm_trim[1])
    lhmsup = np.array_split(lhmsup[0],2)
    
    opinf = np.ndarray.tolist(op_trim[0])
    opinf = np.array_split(opinf[0],2)
    opsup = np.ndarray.tolist(op_trim[1])
    opsup = np.array_split(opsup[0],2)
    
    return lhminf, lhmsup, opinf, opsup


def time_vec(video_path, control_lab):
    ### Creates the time vector for plots
    
    #frames_num, fps
    
    # create video capture object
    data = cv2.VideoCapture(video_path)
  
    # count the number of frames
    frames = int(data.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = data.get(cv2.CAP_PROP_FPS)
  
    if(control_lab == 'lemoh'):
        new_frame = 2*frames

    else: new_frame = frames
    # calculate duration of the video
    seconds = round(frames / fps)
    time_vec = np.linspace(0, seconds, new_frame)
    return time_vec


def displac_axis(neck_x:np.array, neck_y:np.array, mid_hip_x:np.array, mid_hip_y:np.array, L5_x: np.array, L5_y: np.array, position):
    ### COmputes the displacement of a given point
    
    displa_vec_norm = []

    vec1 = np.array(neck_x[position],neck_y[position])
    vec2 = np.array(mid_hip_x[position], mid_hip_y[position])

    displa_max = np.linalg.norm( vec1 - vec2 )
    ## maximum distante between de neck and hip point (each axis)

    neck_points = np.array([neck_x,neck_y])
    mid_hip_points = np.array([mid_hip_x, mid_hip_y])
    L5_points = np.array([L5_x,L5_y])

    for index in range(len(neck_points)):
        displa_frame =  np.linalg.norm(neck_points[index] - L5_points[index])
        displa_vec_norm.append(displa_frame/displa_max)

    return displa_vec_norm


def get_norm_vec(x0, y0, x1, y1):
    ### gets the vector norm


    # d = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    # dt = (2/3)*d
    #ratio_t = #dt/d
    # pair_estm = [(1-ratio_t)*x0 + ratio_t*x1,(1-ratio_t)*y0 + ratio_t*y1] # the estimated point relies 2/3 of the starting point

    vec1 = np.array([x0,y0])
    vec2 = np.array([x1,y1])
    vec_ori = np.subtract(vec2, vec1)
    vec_norm = np.linalg.norm(vec_ori)
    
    return vec_norm


def variance_over_samples(disp_vec:np.array, window_size_half:int, step_size:int, vec_norm:np.float):
    #### Calculates the windowed variance and the std deviation of a given data

    variance_vec = []
    std_dev_vec = []
 
    for item in range(step_size,len(disp_vec), step_size):
 
        if (round(step_size + window_size_half) < len(disp_vec)):
 
            window_vec = disp_vec[(item-window_size_half):(item+window_size_half)]
            std_dev_vec.append(stats.stdev(window_vec))
            variance_vec.append(stats.variance(window_vec))
            # 
# 
        else:
            break

    # variance_vec = stats.stdev(disp_vec)
# 
    # variance_vec_norm = np.multiply(variance_vec, (1/vec_norm))

    return variance_vec, std_dev_vec



def get_point_estim(x0, y0, x1, y1):
    ###function that calculates the x and y points of the estimated point


    # d = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    # dt = (2/3)*d
    #ratio_t = #dt/d
    # pair_estm = [(1-ratio_t)*x0 + ratio_t*x1,(1-ratio_t)*y0 + ratio_t*y1] # the estimated point relies 2/3 of the starting point

    vec1 = np.array([x0,y0])
    vec2 = np.array([x1,y1])
    frac_vec = 2/3

    vec_ori = np.subtract(vec2, vec1)
    vec_part = np.multiply(vec_ori, frac_vec) 
    
    pair_estm = vec1 + vec_part
    return pair_estm

def pop_outlayers_interpol(vector:np.array, threshold_change, window_neighbourhood):
    #remove the outlayers from the acquisition

    for item in range(window_neighbourhood, len(vector)):
        change_over_frames = abs(vector[item] - vector[item-window_neighbourhood])

        if (change_over_frames > threshold_change):
            vector_popped = np.delete(vector, item)

    if 'vector_popped' in locals():
        arr2_interp = interp.interp1d(np.arange(len(vector_popped)), vector_popped)
        interpol_vector = arr2_interp(np.linspace(0,len(vector_popped)-1, len(vector)))
        print("condition satisfied")
        return interpol_vector

    else:
     print('Nothing Changed')
     return vector



def smooth_savgol (vector:np.array, window_size, poly_order, model):
    ### Savitsky GOlay filter to smooth data
    
    wind_num = round(len(vector)/window_size)

    if (wind_num % 2) == 0:
        wind_num = round(len(vector)/window_size) + 1
    
    else:
        wind_num = wind_num

    result_savgol = savgol_filter(vector, wind_num, polyorder=poly_order, mode=model)
    return result_savgol
    

def pixel_for_m(pixel: np.array, fator: float):
    ### Converts the pixel data to meters

    px = np.array(pixel) # array em pixel extraído pelo OpenPose
    cm = fator * px  # fator é a relação encontrada entre pixel e cm
    conversion = cm/100 # Convertendo cm para metros
    return conversion


def get_angle(edge1,  edge2, intersection):
    # assert tuple(sorted(edge1)) in edges
    # assert tuple(sorted(edge2)) in edges
    # assert tuple(sorted(edge3)) in edges
    edge1 = np.array(edge1)
    edge2 = np.array(edge2)
    intersection = np.array(intersection)
    
    # mid_point = edge1.intersection(edge2).pop()

    # a = (np.subtract(edge1,edge2)).pop()
    # b = (np.subtract(edge2,edge1)).pop()

    v1 = np.subtract(edge1,intersection)
    v2 = np.subtract(edge2,intersection)

    angle = (math.degrees(np.arccos(np.dot(v1,v2)
                                    /(np.linalg.norm(v1)*np.linalg.norm(v2)))))
    return angle

    
## Wavelet Filtering
def lowpassfilter(signal, thresh = 0.20, wavelet="db13"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal
