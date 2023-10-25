import cv2
import numpy as np



video_path = '/home/lamic/Openpose-pedro/LEMOH_EXP/arm_abduction_test/cam1.avi'

# reading the video 
source = cv2.VideoCapture(video_path) 

# We need to set resolutions. 
# so, convert them from float to integer. 
frame_width = int(source.get(3)) 
frame_height = int(source.get(4)) 
   
size = (frame_width, frame_height) 

result = cv2.VideoWriter('/home/lamic/Openpose-pedro/LEMOH_EXP/output_output_cam1_lmh_video_equalized.avi',  
            cv2.VideoWriter_fourcc(*'MJPG'), 
            60, size, 0)


result2 = cv2.VideoWriter('/home/lamic/Openpose-pedro/LEMOH_EXP/output_output_cam1_lmh_video_CLAHE.avi',  
            cv2.VideoWriter_fourcc(*'MJPG'), 
            60, size, 0) 


while True: 
  
    # extracting the frames 
    ret, img = source.read() 
      
    # converting to gray-scale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    equalized1 = cv2.equalizeHist(gray)
    
    
    clahe = cv2.createCLAHE(clipLimit=5)
    equalized2 = clahe.apply(gray)


    # write to gray-scale 
    result.write(equalized1)
    result2.write(equalized2)

    # displaying the video 
    #cv2.imshow("Live", equalized) 
  
    # exiting the loop 
    #key = cv2.waitKey(1) 
    #if key == ord("q"): 
    #    break
      
# closing the window 
cv2.destroyAllWindows() 
source.release()