from ImageTracking.ClassifyImages import ClassifyImages
# import the calibration as well
import cv2
import numpy as np
import os
import time
import sys 
from kalman import Kalman
from constants import *

# ---- Get the kalman setup ----- #
kalman_l = Kalman()
kalman_r = Kalman()

# -- some init values for the main script -- #
frame = 0
objectFoundL = False
objectFoundR = False
objectVisibleL = False 
objectVisibleR = False  
track_l = kalman_l.state
track_r = kalman_r.state
sensedLs = []
sensedRs = []
predictedLs = []
predictedRs = []
NaiveBayesL = np.ones(3)   
NaiveBayesR = np.ones(3)   
stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=96,
        blockSize=9,
        P1=477,
        P2=2048,
        disp12MaxDiff=29,
        preFilterCap=1, 
        uniquenessRatio=0,
        speckleWindowSize=0,
        speckleRange=0)
    

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
    seg = seg[250:,350:1200]
    xy = np.where(seg>10)
    x = xy[0].mean() if xy[0].size>0 else np.nan     
    y = xy[1].mean() if xy[1].size>0 else np.nan      
    try:         
        x = int(x)+250
        y = int(y)+350
        border.append([(y-halfBorder,x+halfBorder), (y+halfBorder,x-halfBorder)])
    except Exception:  
        x=-1
        y=-1
    mean = [x,y]
    
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),border,mean

def stereo2Dto3D(left,right, stereo, Q, yPL, xPL):
    #resize for faster computation
    smallL = cv2.resize(left, (0,0), fx=0.5, fy=0.5) 
    smallR = cv2.resize(right, (0,0), fx=0.5, fy=0.5)

    x_px = int(xPL*0.5)
    y_px = int(yPL*0.5)

    #calculate disparity
    disp = stereo.compute(smallL, smallR).astype(np.float32) / 16.0

    #adjust focal lenght according to camera specs
    Q[0,0]=2.45
    Q[1,1]=2.45

    #Calculate 3D points
    depth = cv2.reprojectImageTo3D(disp, Q)
    return depth[x_px][y_px]


def trackAndClassifyObjectInImage(img,img_org,backgroundImg,backgroundImg_org,sensed,predicted,track,kalman,objectFound,NaiveBayes):
     # object first found on right half of 
            mask,border,mean = segmentObject(img, backgroundImg, 250, 720, 350, 1200)
            mask_org,border_org,mean_org = segmentObject(img_org, backgroundImg_org, 250, 720, 350, 1200)

            x, y = mean
            if border == []:
                objectVisible = False
                xP = -1
                yP = -1  
                probs = np.array([0])
                class_ = "Nothing"           
            else:
                objectVisible = True  
            
                if (not objectFound) and (y > 600):
                    objectFound = True

                # count number of times object spotted on left (y<500) of conveyor
                yLvals = [row[1] for row in sensed]

                # test of current object has been spotted to left
                beenLeft = sum(1 for y in yLvals if y < 600) > 5

                # see if object is recently on right
                nowOnRight = sum(yLvals[-5:]) > 5*600

                # move on to next object
                if beenLeft and nowOnRight:
                    sensed = [] # reset the detector
                    predicted = [] # clear old predictions
                    kalman.reset() # reset kalman

                sensed.append(mean) # add current measured state to existing list                   
                class_ = "Nothing"
                if border_org != []:
                    # extract cropped image of object and apply classificaiton algorithm
                    c_img = img_org[mean_org[0]-120:mean_org[0]+120,mean_org[1]-120:mean_org[1]+120,:]
                    #cl = str(model.classify_img(c_img,False))
                    cll = model.classify_img(c_img,False)
                    if mean_org[1] < 1000:
                        # The likelihood of this being a cup
                        NaiveBayes[0] = cll[1][0]*NaiveBayes[0]/(cll[1][0]*NaiveBayes[0]+cll[1][1]*NaiveBayes[1]+cll[1][2]*NaiveBayes[2])
                        # The likelihood of this being a book
                        NaiveBayes[1] = cll[1][1]*NaiveBayes[1]/(cll[1][0]*NaiveBayes[0]+cll[1][1]*NaiveBayes[1]+cll[1][2]*NaiveBayes[2])
                        # The likelihood of this being a box
                        NaiveBayes[2] = cll[1][2]*NaiveBayes[2]/(cll[1][0]*NaiveBayes[0]+cll[1][1]*NaiveBayes[1]+cll[1][2]*NaiveBayes[2])
                        class_ = model.categoryDict[np.argmax(NaiveBayes)]
                    elif mean_org[1] > 1000:
                        NaiveBayes = np.ones(3)
                # show cropped image (of object) if desired
                    if showCrop_L:
                        cv2.imshow("Crop left",c_img)
            
                    if showCrop_R:
                        cv2.imshow("Crop right",c_img)
                

                # Kalman update only when object visible/sensed
                Z = np.array([[x],[y]])
                track = kalman.update(track,Z)

            # Object on converyor
            if objectFound:                
                track = kalman.predict(track)

                xP = track[0][0]
                yP = track[3][0]

                xP = max(0, min(720, xP)) # stateL = [xpos, xvel, xaccel, ypos, yvel, yaccel]^T
                yP = max(0, min(1280, yP))

                track[0][0] = xP
                track[3][0] = yP

                predicted.append([xP, yP])   

                # plot all previous sensed states
                for sense in sensed:
                    x = sense[0]
                    y = sense[1]  
                    cv2.circle(img, (y, x), 2, sensedColor, -1)                   

                # plot all previous predicted states
                for predictes in predicted:
                    x = predictes[0]
                    y = predictes[1]  
                    cv2.circle(img, (int(y), int(x)), 2, predictedColor, -1)   
 
                # plot current preditect state
                cv2.circle(img, (int(yP),int(xP)), 6, predictedColor, -1)

            if objectVisible:
                # plot current sensed state
                xS = mean[0] 
                yS = mean[1] 
                cv2.circle(img, (yS,xS), 6, sensedColor, -1)  
                cv2.rectangle(img, border[0][0], border[0][1], visibleColor, 2)
            else:
                xS,yS = -1,-1
            return sensed,predicted,track,objectFound,objectVisible,xP,yP,xS,yS,NaiveBayes,class_

if __name__ == "__main__":

    model = ClassifyImages(load_model=True,dataset='ImageTracking/dataset.csv',load_path="ImageTracking/model_save")  
    video = runVideoStream() 

    for left,right in video:
        frame += 1
        left_org = left
        right_org = right
        left = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
        right = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

        sensedLs,predictedLs,track_l,objectFoundL,objectVisibleL,xPL,yPL,xSL,ySL,NaiveBayesL,classL = trackAndClassifyObjectInImage(left,left_org,backgroundL,backgroundL_org,sensedLs,predictedLs,track_l,kalman_l,objectFoundL,NaiveBayesL)
        probLs = NaiveBayesL
        probL = NaiveBayesL[np.argmax(NaiveBayesL)]

        sensedRs,predictedRs,track_r,objectFoundR,objectVisibleR,xPR,yPR,xSR,ySR,NaiveBayesR,classR = trackAndClassifyObjectInImage(right,left_org,backgroundR,backgroundL_org,sensedRs,predictedRs,track_r,kalman_r,objectFoundR,NaiveBayesR)
        probRs = NaiveBayesR
        probR = NaiveBayesR[np.argmax(NaiveBayesR)]

        
        
        # vvvvvv Make a pretty images, with information vvvvv #
        # combine left and right info into single choice
        numVisible = 0
        probs = [0, 0, 0]
        if objectVisibleL:
            probs += probLs
            numVisible += 1
        if objectVisibleR:
            probs += probRs
            numVisible += 1 
        if numVisible > 0:
            probs /= numVisible

        if sum(probs) == 3:
            class_ = "Nothing"
        else:
            class_ = categoryDict[np.argmax(probs)] 
        prob = probs[np.argmax(probs)]                


        left_og = left
        right_og = right

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
            elif class_ == "box":
                canvas[10:10+250,1260:1260+250,:] = cv2.imread('box.jpg')

            cv2.putText(canvas, 'Classification: ' + class_, (300,2*bigLineHeight), font, bigFontScale, visibleColor, lineType)
            cv2.putText(canvas, 'Probability: ' + str(round(100*prob,1)) + '%', (300,3*bigLineHeight), font, bigFontScale, visibleColor, lineType)
        if objectVisibleL and objectVisibleR:
            cv2.putText(canvas, '2D Measured: (yL, yR, x)=(' + str(ySL) + ', ' + str(ySR) + ', ' + str(int(round(0.5*(xSL + xSR),0))) + ')', (300,4*bigLineHeight), font, bigFontScale, sensedColor, lineType)                
        if objectFoundL and objectFoundR:
            xP = 0.5*(xPL + xPR)
            yP = 0.5*(yPL + yPR)
            cv2.putText(canvas, '2D Predicted: (yL, yR, x)=(' + str(round(yP,1)) + ', ' + str(round(yP,1)) + ', ' + str(int(round(xP,1))) + ')', (300,5*bigLineHeight), font, bigFontScale, predictedColor, lineType)
            posW=stereo2Dto3D(left_og,right_og, stereo, Q, yP, xP)
            xW,yW,zW = posW[0],posW[1],posW[2]
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
        