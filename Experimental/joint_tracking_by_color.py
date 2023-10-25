# Python code for Color Detection

import numpy as np
import cv2
import math


video_path = '/home/lamic/Openpose-pedro/LEMOH_EXP/abduction_PedroTrial_Back_grayscale_output.avi'
video_name = video_path.split('/')[-1]
# Create point matrix get coordinates of mouse click on image
point_matrix = np.zeros((2,2),np.int)

counter = 0
def mousePoints(event,x,y,flags,params):
    global counter
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[counter] = x,y
        counter = counter + 1


# Capturing video through cap
cap = cv2.VideoCapture(video_path)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES,0) # takes frame 0
_, frame = cap.read()

frame_name = video_name + '_frame0.jpg'
cv2.imwrite(frame_name,frame) 
# Read frame
img = cv2.imread(frame_name)
 


########################## selecting first ROI for the detector #################################

# ROI = cv2.selectROI('Select the ROI of the joint', img)
# cropped_frame = img[int(ROI[1]):int(ROI[1]+ROI[3]), 
#                       int(ROI[0]):int(ROI[0]+ROI[2])]


while True:
   for x in range (0,2):
       cv2.circle(img,(point_matrix[x][0],point_matrix[x][1]),3,(0,255,0),cv2.FILLED)

   if counter == 2:
       starting_x = point_matrix[0][0]
       starting_y = point_matrix[0][1]

       ending_x = point_matrix[1][0]
       ending_y = point_matrix[1][1]
       # Draw rectangle for area of interest
       cv2.rectangle(img, (starting_x, starting_y), (ending_x, ending_y), (0, 255, 0), 3)

       # Cropping image
       cropped_frame = img[starting_y:ending_y, starting_x:ending_x]
       cv2.imshow("ROI", cropped_frame)

   # Showing original image
   cv2.imshow("Original Image ", img)
   # Mouse click event on original image
   cv2.setMouseCallback("Original Image ", mousePoints)

   # Refreshing window all time
   if cv2.waitKey(10) & 0xFF == ord('q'):
       cv2.destroyAllWindows()
       break


# Printing updated point matrix
print(point_matrix)

############################ Joint detection based on Color ####################################### 

stopper = 0

x_joint = []
y_joint = []



def find_joint (cropped_frame):
    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

    # Set range for green color and
    # define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    green_mask = cv2.erode(green_mask, None, iterations=2)
    green_mask = cv2.dilate(green_mask, None, iterations=2)

        
    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernel = np.ones((5, 5), "uint8")
    
    # For green color
    green_mask = cv2.dilate(green_mask, kernel)
    res_green = cv2.bitwise_and(cropped_frame, cropped_frame,
                                mask = green_mask)	

    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


Fst_contours, Fst_hierarchy = find_joint(cropped_frame)

for pic, contour in enumerate(Fst_contours):
        area = cv2.contourArea(contour)
        if(area > 200):
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x1 + starting_x - w1 , y1 + starting_y - h1 ),
                                    ( starting_x + x1 + w1, starting_y + y1 + h1),
                                    (255, 0, 255), 2)
            
            #cv2.putText(img, "Green Colour", (x1, y1),	cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))


x, y, w, h = x1,y1,w1,h1 # set first values before update

width, height = 40,40 ## size of Bounding Box futher on

reference_x, reference_y = starting_x, starting_y ## loads reference to original image from first frame

p_f0 = [x + reference_x + w/2, y + reference_y + h/2] #center point of 1st detection
p = p_f0



while(stopper<=(num_frames)):
    # Reading the video from the
    # cap in image frames
    _, imageFrame = cap.read()

    next_ROI = imageFrame[ int(p[1] - height/2) : int(p[1] + height/2), ## Y all along
                             int(p[0] - width/2) : int(p[0] + width/2) ]   ## X all along


    contours, hierarchy = find_joint(next_ROI)


    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 100):
            x, y, w, h = cv2.boundingRect(contour)
            p = [ int(x + reference_x + w/2), int(y + reference_y + h/2) ] ## center point of  detection
            img = cv2.rectangle(imageFrame, (int(p[0] - w/2), int(p[1] - h/2)), #top left of rect
                                    (int(p[0] + w/2), int(p[1] + h/2)),          #bottom right of rect
                                    (255, 0, 255), 2)
            
            #cv2.putText(imageFrame, "Green Colour", (x, y),	cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            
                        
            #top_left = [int(p[0] - width/2), int(p[1] - height/2)]
            #bottom_right = [int(p[0] + width/2), int(p[1] + height/2)]

    reference_x, reference_y = [ int(p[0] - w/2) , int(p[1] - h/2) ]


    x_joint.insert(stopper, x)
    y_joint.insert(stopper, y)


    stopper = stopper+1
     
    
    # Program Termination
    resized = cv2.resize(imageFrame, (540,960))
    cv2.imshow("Multiple Color Detection in Real-TIme", resized)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
