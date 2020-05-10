from ImageTracking.ClassifyImages import ClassifyImages
# import the calibration as well
import cv2
import numpy as np
import os
import time
import sys 

test = False
Occlusion = False

showL = True
showR = False
showLc = False
showRc = False   



dataset = "/Users/oliverbehrens/Desktop/PerceptionFinal/Perception/src/ImageTracking/dataset.csv"
load_path = "/Users/oliverbehrens/Desktop/PerceptionFinal/Perception/src/ImageTracking/model_save"
data1 = np.load('/Users/oliverbehrens/Desktop/PerceptionFinal/Perception/src/CameraCalibration/calib_stereo.npz')
data2 = np.load('/Users/oliverbehrens/Desktop/PerceptionFinal/Perception/src/CameraCalibration/rectified_stereo.npz')
path_left_No_Occlusion = "/Users/oliverbehrens/Desktop/left"
path_right_No_Occlusion = "/Users/oliverbehrens/Desktop/right"
path_left_withOcclusion  = "/Users/oliverbehrens/Desktop/Stereo_conveyor_with_occlusions/left"
path_right_withOcclusion  = "/Users/oliverbehrens/Desktop/Stereo_conveyor_with_occlusions/right"
backgroundLNoOcclusion = "/Users/oliverbehrens/Desktop/left/1585434282_261431932_Left.png"
backgroundRNoOcclusion = "/Users/oliverbehrens/Desktop/right/1585434282_261431932_Right.png"
backgroundLwithOcclusion = "/Users/oliverbehrens/Desktop/Stereo_conveyor_with_occlusions/left/1585434771_711674690_Left.png"
backgroundRwithOcclusion = "/Users/oliverbehrens/Desktop/Stereo_conveyor_with_occlusions/right/1585434771_711674690_Right.png"

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

cameraMatrix1 = data1['cameraMatrix1']
cameraMatrix2 = data1['cameraMatrix2']
distCoeffs1 = data1['distCoeffs1']
distCoeffs2 = data1['distCoeffs2']

R1 = data2['R1']
R2 = data2['R2']
P1 = data2['P1']
P2 = data2['P2']

img = cv2.imread(backgroundLNoOcclusion)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray.shape[::-1], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray.shape[::-1], cv2.CV_32FC1)

backgroundLNoOcclusion = cv2.remap(cv2.imread(backgroundLNoOcclusion), map1x, map1y, cv2.INTER_LINEAR);
backgroundRNoOcclusion = cv2.remap(cv2.imread(backgroundRNoOcclusion), map2x, map2y, cv2.INTER_LINEAR);
backgroundLwithOcclusion = cv2.remap(cv2.imread(backgroundLwithOcclusion), map1x, map1y, cv2.INTER_LINEAR);
backgroundRwithOcclusion = cv2.remap(cv2.imread(backgroundRwithOcclusion), map2x, map2y, cv2.INTER_LINEAR);

categoryDict = {0:"cup",1:"book",2:"box",3:"Nothing"}

#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Kalman Additions ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

def update(x, P, Z, H, R):
    Y = Z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.pinv(S)
    X_new = x + K @ Y
    P_new = (I - K @ H) @ P
    return X_new, P_new
    
def predict(x, P, F, u):
    X_new = F @ x + u
    P_new = F @ P @ F.T
    return X_new, P_new   
    
# Initial State (in left image) 
state = np.array([[0],  # x pos
                  [0],  # x vel
                  [0],  # x accel
                  [0],  # y pos
                  [0],  # y vel
                  [0]]) # y accel

# Initial Uncertainty (assumed large)
P = np.array([[1000, 0, 0, 0, 0, 0],
              [0, 1000, 0, 0, 0, 0],
              [0, 0, 1000, 0, 0, 0],
              [0, 0, 0, 1000, 0, 0],
              [0, 0, 0, 0, 1000, 0],
              [0, 0, 0, 0, 0, 1000]])

# External motion (assumed environment/camera is stationary)
u = np.array([[0],[0],[0],[0],[0],[0]])

# Transition matrix (x = x0 + vt + 1/2 at^2, etc. )
F = np.array([[1, 1, 0.5, 0, 0,   0],
              [0, 1,   1, 0, 0,   0],
              [0, 0,   1, 0, 0,   0],
              [0, 0,   0, 1, 1, 0.5],
              [0, 0,   0, 0, 1,   1],
              [0, 0,   0, 0, 0,   1]])

# Observation matrix (only get measured position)
H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

# Measurement Uncertainty
R = np.array([[1, 0], [0, 1]])

# Identity Matrix
I = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Kalman Additions ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

def runVideoStream():
    # Get the number of images in the left folder (The Video)
    # There is the same number of images in both left and right folder.
    left_pics = sorted([f for f in os.listdir(path_left)])
    right_pics = sorted([f for f in os.listdir(path_right)])
    num_pics = len(left_pics)
    
    # Here we can now make use of what is known as generators
    # This creates an object, where (for each time we call it)
    # We get the next element of the loop. The generator is created
    # using yield, instead of return

    for i in range(num_pics):
        left  = cv2.imread(path_left+"/"+left_pics[i])
        right = cv2.imread(path_right+"/"+right_pics[i])
        yield left,right

    pass
     
def TestConvSeg():
    path = "C:\\Users\\swang\\Desktop\\ConvTestImg"
    pics = [f for f in os.listdir(path)]
    for i in range(len(pics)):
        img  = cv2.imread(path+"/"+pics[i])
        yield img

def segmentObject(img, bg, x1, x2, y1, y2):
    halfBorder = 160
    image = img.copy()
    background = bg.copy()
    #Complete removal of backgroubd
    image = background-image
    image[image>200] = 0
    image[image<50] = 0
    erode_ = np.ones((6,6),np.uint8)
    dialte_ = np.ones((10,10),np.uint8)
    image = cv2.erode(image,erode_)
    image = cv2.dilate(image,dialte_)
    border = []
    mean = []
    #locate the object in the image
    seg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    seg = seg[x1:x2,y1:y2]
    xy = np.where(seg>10)
    x = xy[0].mean() if xy[0].size>0 else np.nan     
    y = xy[1].mean() if xy[1].size>0 else np.nan      
    try:         
        x = int(x)+x1
        y = int(y)+y1
        border.append([(y-halfBorder,x+halfBorder), (y+halfBorder,x-halfBorder)])
    except Exception:  
        x=-1
        y=-1
    mean = [x,y]
    
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),border,mean

if __name__ == "__main__":

    model = ClassifyImages(load_model=True, dataset=dataset, load_path=load_path)
        
    if not test:
        
        if Occlusion:
            path_left = path_left_withOcclusion
            backgroundL = backgroundLwithOcclusion
            path_right = path_right_withOcclusion
            backgroundR = backgroundRwithOcclusion
        else:
            path_left = path_left_No_Occlusion
            backgroundL = backgroundLNoOcclusion
            path_right = path_right_No_Occlusion
            backgroundR = backgroundRNoOcclusion
        
        video = runVideoStream() 

        frame = 0
        objectFoundL = False
        objectFoundR = False
        objectVisibleL = False 
        objectVisibleR = False  
        stateL = state
        stateR = state        
        PL = P
        PR = P
        sensedLs = []
        sensedRs = []
        predictedLs = []
        predictedRs = []        

        for left,right in video:
            frame += 1

            left = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR);
            right = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR);

            # attempt to locate object in image w/background removed
            x1, x2, y1, y2 = 250, 720, 350, 1200
            maskL,borderL,meanL = segmentObject(left, backgroundL, x1, x2, y1, y2)

            x1, x2, y1, y2 = 250, 720, 350, 1200
            maskR,borderR,meanR = segmentObject(right, backgroundR, x1, x2, y1, y2)

            # center of located object or (-1, -1) if no object found
            xL, yL = meanL
            xR, yR = meanR            

            # No object visible/sensed
            if borderL == []:
                objectVisibleL = False              

            # Object visible/sensed
            else:
                objectVisibleL = True              
                
                # object first found on right half of conveyor
                if (not objectFoundL) and (yL > 600):
                    objectFoundL = True

                # count number of times object spotted on left (y<500) of conveyor
                yLvals = [row[1] for row in sensedLs]

                # test of current object has been spotted to left
                beenLeftL = sum(1 for y in yLvals if y < 600) > 5

                # see if object is recently on right
                nowOnRightL = sum(yLvals[-5:]) > 5*600

                # move on to next object
                if beenLeftL and nowOnRightL:
                    stateL = state      # reset initial state
                    PL = P              # reset uncertainty
                    sensedLs = []       # reset sensed states
                    predictedLs = []    # reset predicted states             

                sensedLs.append(meanL) # add current measured state to existing list                   

                # extract cropped image of object and apply classificaiton algorithm
                masked_left = cv2.bitwise_and(left,left,mask=maskL)
                c_imgL = masked_left[meanL[0]-120:meanL[0]+120,meanL[1]-120:meanL[1]+120,:]
                _,probLs,_ = model.classify_img(c_imgL,False) 
                # print(probLs)
                classL = categoryDict[np.argmax(probLs)] 
                probL = probLs[np.argmax(probLs)]               

                # show cropped image (of object) if desired
                if showLc:
                    cv2.imshow("Crop",c_imgL)

                # Kalman update only when object visible/sensed
                ZL = np.array([[xL],[yL]])
                stateL,PL = update(stateL,PL,ZL,H,R)

            # Object on converyor
            if objectFoundL:                
                stateL,PL = predict(stateL,PL,F,u)

                xPL = stateL[0][0]
                yPL = stateL[3][0]

                xPL = max(0, min(720, xPL)) # stateL = [xpos, xvel, xaccel, ypos, yvel, yaccel]^T
                yPL = max(0, min(1280, yPL))

                stateL[0][0] = xPL
                stateL[3][0] = yPL

                predictedLs.append([xPL, yPL])   

                # plot all previous sensed states
                for sensed in sensedLs:
                    xL = sensed[0]
                    yL = sensed[1]  
                    cv2.circle(left, (yL, xL), 2, sensedColor, -1)                   

                # plot all previous predicted states
                for predicted in predictedLs:
                    xL = predicted[0]
                    yL = predicted[1]  
                    cv2.circle(left, (int(yL), int(xL)), 2, predictedColor, -1)   
 
                # plot current preditect state
                cv2.circle(left, (int(yPL),int(xPL)), 6, predictedColor, -1)

            if objectVisibleL:
                # plot current sensed state
                xSL = meanL[0] 
                ySL = meanL[1] 
                cv2.circle(left, (ySL,xSL), 6, sensedColor, -1)  
                cv2.rectangle(left, borderL[0][0], borderL[0][1], visibleColor, 2) 

            # No object visible/sensed
            if borderR == []:
                objectVisibleR = False              

            # Object visible/sensed
            else:
                objectVisibleR = True              
                
                # object first found on right half of conveyor
                if (not objectFoundR) and (yR > 600):
                    objectFoundR = True

                # count number of times object spotted on left (y<500) of conveyor
                yRvals = [row[1] for row in sensedRs]

                # test of current object has been spotted to left
                beenLeftR = sum(1 for y in yRvals if y < 600) > 5

                # see if object is recently on right
                nowOnRightR = sum(yRvals[-5:]) > 5*600

                # move on to next object
                if beenLeftR and nowOnRightR:
                    stateR = state      # reset initial state
                    PR = P              # reset uncertainty
                    sensedRs = []       # reset sensed states
                    predictedRs = []    # reset predicted states             

                sensedRs.append(meanR) # add current measured state to existing list                   

                # extract cropped image of object and apply classificaiton algorithm
                masked_right = cv2.bitwise_and(right,right,mask=maskR)
                c_imgR = masked_right[meanR[0]-80:meanR[0]+80,meanR[1]-80:meanR[1]+80,:]
                _,probRs,_ = model.classify_img(c_imgR,False) 
                # print(probLs)
                classR = categoryDict[np.argmax(probRs)] 
                probR = probRs[np.argmax(probRs)]               

                # show cropped image (of object) if desired
                if showRc:
                    cv2.imshow("Crop",c_imgR)

                # Kalman update only when object visible/sensed
                ZR = np.array([[xR],[yR]])
                stateR,PR = update(stateR,PR,ZR,H,R)

            # Object on converyor
            if objectFoundR:                
                stateR,PR = predict(stateR,PR,F,u)

                xPR = stateR[0][0]
                yPR = stateR[3][0]

                xPR = max(0, min(720, xPR)) # stateL = [xpos, xvel, xaccel, ypos, yvel, yaccel]^T
                yPR = max(0, min(1280, yPR))

                stateR[0][0] = xPR
                stateR[3][0] = yPR

                predictedRs.append([xPR, yPR])     

                # plot all previous sensed states
                for sensed in sensedRs:
                    xR = sensed[0]
                    yR = sensed[1]  
                    cv2.circle(right, (yR, xR), 2, sensedColor, -1)                   

                # plot all previous predicted states
                for predicted in predictedRs:
                    xR = predicted[0]
                    yR = predicted[1]  
                    cv2.circle(right, (int(yR), int(xR)), 2, predictedColor, -1)   
 
                # plot current preditect state
                cv2.circle(right, (int(yPR),int(xPR)), 6, predictedColor, -1)

            if objectVisibleR:
                # plot current sensed state
                xSR = meanR[0] 
                ySR = meanR[1] 
                cv2.circle(right, (ySR,xSR), 6, sensedColor, -1)  
                cv2.rectangle(right, borderR[0][0], borderR[0][1], visibleColor, 2) 

            # combine left and right info into single choice
            numVisible = 0
            probs = [0, 0, 0, 0]
            if objectVisibleL:
                probs += probLs
                numVisible += 1
            if objectVisibleR:
                probs += probRs
                numVisible += 1 
            if numVisible > 0:
                probs /= numVisible

            class_ = categoryDict[np.argmax(probs)] 
            prob = probs[np.argmax(probs)]                

            # combine images and annotations
            left = cv2.resize(left, (960, 540), interpolation = cv2.INTER_AREA)
            right = cv2.resize(right, (960, 540), interpolation = cv2.INTER_AREA)                
            sideBySide = np.concatenate((left, right), axis=1)
            canvas = np.zeros((1080,1920,3), dtype=np.uint8)
            canvas[270:270+540,0:1920,:] = sideBySide
            
            cv2.rectangle(canvas, (0,270), (1920,270+540), defaultColor, 6)
            cv2.line(canvas, (960,270), (960,1080), defaultColor, 6)    
            cv2.putText(canvas, 'Frame: ' + str(frame), (300,bigLineHeight), font, bigFontScale, defaultColor, lineType)
            if objectVisibleL or objectVisibleR:
                if class_ == 'cup':
                    canvas[10:10+250,1260:1260+250,:] = cv2.imread('cup.jpg')
                elif class_ == 'book':
                    canvas[10:10+250,1260:1260+250,:] = cv2.imread('book.jpg')
                else:
                    canvas[10:10+250,1260:1260+250,:] = cv2.imread('box.jpg')

                cv2.putText(canvas, 'Classification: ' + class_, (300,2*bigLineHeight), font, bigFontScale, visibleColor, lineType)
                cv2.putText(canvas, 'Probability: ' + str(round(100*prob,1)) + '%', (300,3*bigLineHeight), font, bigFontScale, visibleColor, lineType)
            if objectVisibleL and objectVisibleR:
                cv2.putText(canvas, '2D Measured: (yL, yR, x)=(' + str(ySL) + ', ' + str(ySR) + ', ' + str(int(round(0.5*(xSL + xSR),0))) + ')', (300,4*bigLineHeight), font, bigFontScale, sensedColor, lineType)                
            if objectFoundL and objectFoundR:
                xP = 0.5*(xPL + xPR)
                cv2.putText(canvas, '2D Predicted: (yL, yR, x)=(' + str(round(yPL,1)) + ', ' + str(round(yPR,1)) + ', ' + str(int(round(xP,1))) + ')', (300,5*bigLineHeight), font, bigFontScale, predictedColor, lineType)
                xW,yW,zW = 12.3,45.6,78.9
                # xW,yW,zW = stereo2Dto3D(yPL,yPR, xP)
                cv2.putText(canvas, '3D Predicted: (xW, yW, zW)=(' + str(round(xW,1)) + ', ' + str(round(yW,1)) + ', ' + str(round(zW,1)) + ')', (300,6*bigLineHeight), font, bigFontScale, worldColor, lineType)                
            
            cv2.putText(canvas, 'Left Undistorted and Rectified', (300,875+smallLineHeight), font, smallFontScale, defaultColor, lineType)
            if objectVisibleL:
                # Indicate current classification and probability
                cv2.putText(canvas, 'Classification: ' + classL, (300,875+2*smallLineHeight), font, smallFontScale, visibleColor, lineType)
                cv2.putText(canvas, 'Probability: ' + str(round(100*probL,1)) + '%', (300,875+3*smallLineHeight), font, smallFontScale, visibleColor, lineType)
                cv2.putText(canvas, 'Measured: (' + str(ySL) + ', ' + str(xSL) + ')', (300,875+4*smallLineHeight), font, smallFontScale, sensedColor, lineType)   
            if objectFoundL:   
                x = int(270+xPL*960/1280)
                cv2.line(canvas, (0,x), (960,x), predictedColor, 1)                  
                cv2.putText(canvas, 'Predicted: (' + str(round(yPL,1)) + ', ' + str(round(xPL,1)) + ')', (300,875+5*smallLineHeight), font, smallFontScale, predictedColor, lineType)                                              

            cv2.putText(canvas, 'Right Undistorted and Rectified', (960+300,875+smallLineHeight), font, smallFontScale, defaultColor, lineType)
            if objectVisibleR:
                # Indicate current classification and probability
                cv2.putText(canvas, 'Classification: ' + classR, (960+300,875+2*smallLineHeight), font, smallFontScale, visibleColor, lineType)
                cv2.putText(canvas, 'Probability: ' + str(round(100*probR,1)) + '%', (960+300,875+3*smallLineHeight), font, smallFontScale, visibleColor, lineType)
                cv2.putText(canvas, 'Measured: (' + str(ySR) + ', ' + str(xSR) + ')', (960+300,875+4*smallLineHeight), font, smallFontScale, sensedColor, lineType)   
            if objectFoundR:  
                x = int(270+xPR*960/1280)
                cv2.line(canvas, (960,x), (1920,x), predictedColor, 1)                               
                cv2.putText(canvas, 'Predicted: (' + str(round(yPR,1)) + ', ' + str(round(xPR,1)) + ')', (960+300,875+5*smallLineHeight), font, smallFontScale, predictedColor, lineType)

            if objectFoundL and objectFoundR:  
                x = int(270+0.5*(xPL + xPR)*960/1280)
                yL = int(yPL*960/1280)
                yR = int(960 + yPR*960/1280)
                cv2.circle(canvas, (yL,x), 6, worldColor, -1)
                cv2.circle(canvas, (yR,x), 6, worldColor, -1) 
                cv2.line(canvas, (0,x), (1920,x), worldColor, 2)                               

            cv2.imshow("Side By Side",canvas)

            cv2.waitKey(1)
            #cv2.destroyAllWindows()
    else:

        test = TestConvSeg()
        for img in test:
            img = primeImgForClassification(img,True)
            class_ = model.classify_img(img,False)
            print(class_)
            sys.stdout.flush()
            cv2.imshow("image",img)
            k = cv2.waitKey(1000)
            if k == 27:
                cv2.destroyAllWindows()

        #cv2.imshow("background",background)
        #cv2.imshow("test",test_img)
        #primeImgForClassification(test_img)
        #k = cv2.waitKey(0)
        #if k == 27:
        #    cv2.destroyAllWindows()
         