import numpy as np
import cv2 
# -- some debugging arguments -- #
test = False
showL = True
showR = False
showCrop_L = True
showCrop_R = False  

# -- Choose to test classification and kalman on Occlusion or not -- #
Occlusion = False

 


# --- This loads the camera calibrations --- #
data1 = np.load('CameraCalibration/calib_stereo.npz')
data2 = np.load('CameraCalibration/rectified_stereo.npz')

# --- The path to the video ---- #
path_left_No_Occlusion = "C:\\Users\\swang\\Desktop\\Video\\NoOcclusions\\left"
path_right_No_Occlusion = "C:\\Users\\swang\\Desktop\\Video\\NoOcclusions\\right"
path_left_withOcclusion  = "C:\\Users\\swang\\Desktop\\Video\\Occlusions\\left"
path_right_withOcclusion  = "C:\\Users\\swang\\Desktop\\Video\\Occlusions\\right"

# ---- The static background images ---- #
backgroundLNoOcclusion = path_left_No_Occlusion+"/1585434280_967102051_Left.png"
backgroundRNoOcclusion = path_right_No_Occlusion+"/1585434280_967102051_Right.png"
backgroundLwithOcclusion = path_left_withOcclusion+"/1585434751_268014669_Left.png"
backgroundRwithOcclusion = path_right_withOcclusion+"/1585434751_268014669_Right.png"

# ---- The font with which we will write on the video ----- #
font                   = cv2.FONT_HERSHEY_SIMPLEX
tabWidth               = 15
smallLineHeight        = 28
bigLineHeight          = 38
smallFontScale         = 0.8
bigFontScale           = 1.0
lineType               = 1

defaultColor           = (255,255,255)
visibleColor           = (128,255,255)
sensedColor            = (128,255,0)
predictedColor         = (128,0,255)
worldColor             = (255,128,0)

# ------ The camera matrices ----- #
cameraMatrix1 = data1['cameraMatrix1']
cameraMatrix2 = data1['cameraMatrix2']
distCoeffs1 = data1['distCoeffs1']
distCoeffs2 = data1['distCoeffs2']
R1 = data2['R1']
R2 = data2['R2']
P1 = data2['P1']
P2 = data2['P2']

# --- undistord and rectify background img no occlusion --- #
img = cv2.imread(backgroundLNoOcclusion)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

map1x_noOc, map1y_noOc = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray.shape[::-1], cv2.CV_32FC1)
map2x_noOc, map2y_noOc = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray.shape[::-1], cv2.CV_32FC1)

backgroundLNoOcclusion = cv2.remap(cv2.imread(backgroundLNoOcclusion), map1x_noOc, map1y_noOc, cv2.INTER_LINEAR)
backgroundRNoOcclusion = cv2.remap(cv2.imread(backgroundRNoOcclusion), map2x_noOc, map2y_noOc, cv2.INTER_LINEAR)

# --- undistord and rectify background img no occlusion --- #
img = cv2.imread(backgroundLwithOcclusion)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

map1x_withOc, map1y_withOc = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray.shape[::-1], cv2.CV_32FC1)
map2x_withOc, map2y_withOc = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray.shape[::-1], cv2.CV_32FC1)


backgroundLwithOcclusion = cv2.remap(cv2.imread(backgroundLwithOcclusion), map1x_withOc, map1y_withOc, cv2.INTER_LINEAR)
backgroundRwithOcclusion = cv2.remap(cv2.imread(backgroundRwithOcclusion), map2x_withOc, map2y_withOc, cv2.INTER_LINEAR)

# -- the class category for printing -- #
categoryDict = {0:"cup",1:"book",2:"box"}

# -- choose if we want occlusion or not -- #
if Occlusion:
    path_left = path_left_withOcclusion
    backgroundL = backgroundLwithOcclusion
    path_right = path_right_withOcclusion
    backgroundR = backgroundRwithOcclusion
    map1x = map1x_withOc
    map2x = map2x_withOc
    map1y = map1y_withOc
    map2y = map2y_withOc
else:
    path_left = path_left_No_Occlusion
    backgroundL = backgroundLNoOcclusion
    path_right = path_right_No_Occlusion
    backgroundR = backgroundRNoOcclusion
    map1x = map1x_noOc
    map2x = map2x_noOc
    map1y = map1y_noOc
    map2y = map2y_noOc