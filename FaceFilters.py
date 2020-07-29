import cv2
from imutils.video import VideoStream
import numpy as np
import dlib
from math import hypot
from imutils import face_utils, translate, resize
from scipy.spatial import distance as dist  
from scipy.spatial import ConvexHull
from datetime import datetime


video_capture = cv2.VideoCapture(0)

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  
nose_image = cv2.imread("Pig_Nose.png")
eye_image = cv2.imread("EyePop.png",-1)  
# Define the codec and create VideoWriter object.
#The output is stored in 'video<datetime>.avi' file.

FULL_POINTS = list(range(0, 68))  
FACE_POINTS = list(range(17, 68))  
JAWLINE_POINTS = list(range(0, 17))  
RIGHT_EYEBROW_POINTS = list(range(17, 22))  
LEFT_EYEBROW_POINTS = list(range(22, 27))  
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68))  
 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)  

filters = ["pignose", "eyepop"]
curr_filter = filters[0]
counter = 0

def eye_size(eye):  
   eyeWidth = dist.euclidean(eye[0], eye[3])  
   hull = ConvexHull(eye)  
   eyeCenter = np.mean(eye[hull.vertices, :], axis=0)  
   
   eyeCenter = eyeCenter.astype(int)  
   
   return int(eyeWidth), eyeCenter

def place_eye(frame, eyeCenter, eyeSize):  
   eyeSize = int(eyeSize * 1.5)  
   
   x1 = int(eyeCenter[0,0] - (eyeSize/2))  
   x2 = int(eyeCenter[0,0] + (eyeSize/2))  
   y1 = int(eyeCenter[0,1] - (eyeSize/2))  
   y2 = int(eyeCenter[0,1] + (eyeSize/2))  
   
   h, w = frame.shape[:2]  
   
   # check for clipping  
   if x1 < 0:  
     x1 = 0  
   if y1 < 0:  
     y1 = 0  
   if x2 > w:  
     x2 = w  
   if y2 > h:  
     y2 = h  
   
   # re-calculate the size to avoid clipping  
   eyeOverlayWidth = x2 - x1  
   eyeOverlayHeight = y2 - y1  
   
   # calculate the masks for the overlay  
   eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
   mask = cv2.resize(orig_mask, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
   mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
   
   # take ROI for the verlay from background, equal to size of the overlay image  
   roi = frame[y1:y2, x1:x2]  
   
   # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.  
   roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)  
   
   # roi_fg contains the image pixels of the overlay only where the overlay should be  
   roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask = mask)  
   
   # join the roi_bg and roi_fg  
   dst = cv2.add(roi_bg,roi_fg)  
   
   # place the joined image, saved to dst back over the original image  
   frame[y1:y2, x1:x2] = dst  
    
while True:

    # read a frame from webcam, resize to be smaller
    _, frame = video_capture.read()
    frame = resize(frame, width=800)
    # Capture frame-by-frame


    # the detector and predictor expect a grayscale image
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_frame, 0)

    if curr_filter == "pignose":
        faces = detector(frame)  
        for face in faces:
           landmarks = predictor(gray_frame, face)
            
           top_nose= (landmarks.part(29).x, landmarks.part(29).y)
           center_nose = (landmarks.part(30).x, landmarks.part(30).y)
           left_nose= (landmarks.part(31).x, landmarks.part(31).y)
           right_nose= (landmarks.part(35).x, landmarks.part(35).y)
           
           nose_width =int( hypot(left_nose[0] - right_nose[0],
                              left_nose[1] - right_nose[1]) * 1.7)
           nose_height = int(nose_width * 0.77)
           
           #New nose position
           top_left = (int(center_nose[0] - nose_width / 2),
                                 int(center_nose[1] - nose_height / 2))
           bottom_right = (int(center_nose[0] + nose_width / 2),
                         int(center_nose[1] + nose_height / 2))
           
           
           #Adding the new nose
           nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
           nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
           _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
           
           nose_area = frame[top_left[1]: top_left[1] + nose_height,
                       top_left[0]: top_left[0] + nose_width]
           nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
           final_nose = cv2.add(nose_area_no_nose, nose_pig)
           #cv2.circle(frame, top_nose, 3, (255, 0, 0), -1)
           
           frame[top_left[1]: top_left[1] + nose_height,
                       top_left[0]: top_left[0] + nose_width] = final_nose
       
    if curr_filter == "eyepop":
         # Load the image to be used as our overlay  
        imgEye = eye_image
           
        # Create the mask from the overlay image  
        orig_mask = imgEye[:,:,3]  
           
        # Create the inverted mask for the overlay image  
        orig_mask_inv = cv2.bitwise_not(orig_mask)  
           
        # Convert the overlay image image to BGR  
        # and save the original image size  
        imgEye = imgEye[:,:,0:3]  
        origEyeHeight, origEyeWidth = imgEye.shape[:2]
        
        #nrects = detector(gray_frame, 0)  
   
        for rect in rects:  
            x = rect.left()  
            y = rect.top()  
            x1 = rect.right()  
            y1 = rect.bottom()  
       
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])  
       
            left_eye = landmarks[LEFT_EYE_POINTS]  
            right_eye = landmarks[RIGHT_EYE_POINTS]
            
            leftEyeSize, leftEyeCenter = eye_size(left_eye)  
            rightEyeSize, rightEyeCenter = eye_size(right_eye)  
   
            place_eye(frame, leftEyeCenter, leftEyeSize)  
            place_eye(frame, rightEyeCenter, rightEyeSize)
           
    # Display the resulting frame
    cv2.imshow('Video', frame)


    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

    if key == ord("n"):
        counter = (counter + 1) % len(filters)
        curr_filter = filters[counter]
        
  

# When everything is done, release the capture
cv2.destroyAllWindows()
video_capture.release()
