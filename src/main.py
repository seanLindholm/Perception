from ImageTracking.ClassifyImages import ClassifyImages
# import the calibration as well
import cv2
import numpy as np
import os
import time
import sys 

background = cv2.imread("C:\\Users\\swang\\Desktop\\Video\\NoOcclusions\\left\\1585434282_261431932_Left.png")
background = background[250:,300:1200,:]



def runVideoStreamNoOcculison():
    # Get the number of images in the left folder (The Video)
    # There is the same number of images in both left and right folder.
    path_left = "C:\\Users\\swang\\Desktop\\Video\\NoOcclusions\\left"
    path_right = "C:\\Users\\swang\\Desktop\\Video\\NoOcclusions\\right"
    
    left_pics = [f for f in os.listdir(path_left)]
    right_pics = [f for f in os.listdir(path_right)]
    num_pics = len(left_pics)
    
    # Here we can now make use of what is known as generators
    # This creates and object, where (for each time we call it)
    # We get the next element of the loop. The generator is created
    # using yield, instead of return

    for i in range(num_pics):
        left  = cv2.imread(path_left+"/"+left_pics[i])
        right = cv2.imread(path_right+"/"+right_pics[i])
        yield left,right

    pass

def runVideoStreamWithOccultions():
     # Get the number of images in the left folder (The Video)
    # There is the same number of images in both left and right folder.
    path_left = "C:\\Users\swang\Desktop\Video\Occlusions\left"
    path_right = "C:\\Users\swang\Desktop\Video\Occlusions\\right"
    
    left_pics = [f for f in os.listdir(path_left)]
    right_pics = [f for f in os.listdir(path_right)]
    num_pics = len(left_pics)
    
    # Here we can now make use of what is known as generators
    # This creates and object, where (for each time we call it)
    # We get the next element of the loop. The generator is created
    # using yield, instead of return

    for i in range(num_pics):
        left  = cv2.imread(path_left+"/"+left_pics[i])
        right = cv2.imread(path_right+"/"+right_pics[i])
        yield left,right
    pass
     
def TestConvSeq():
    path = "C:\\Users\\swang\\Desktop\\ConvTestImg"
    pics = [f for f in os.listdir(path)]
    for i in range(len(pics)):
        img  = cv2.imread(path+"/"+pics[i])
        yield img

def primeImgForClassification(img,grayFirst=True):
    if grayFirst:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        gray2 = background.copy()
    
    #resize
    return cv2.resize(gray,(200,200))

if __name__ == "__main__":
    test = True
    model = ClassifyImages(load_model=True,dataset = "./ImageTracking/dataset.csv", load_path="./ImageTracking/model_save")
        
    if not test:
        
        model = ClassifyImages(load_model=True,dataset = "./ImageTracking/dataset.csv", load_path="./ImageTracking/model_save")
        
        video = runVideoStreamNoOcculison()

        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        frame = 0
        for left,right in video:
            frame += 1
            classImg = left[250:,300:1200,:]
            classImg = primeImgForClassification(classImg,True)
            class_ = model.classify_img(classImg,False)

            cv2.putText(left,class_ + " - " + str(frame), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

            cv2.imshow("left",left)
            cv2.imshow("image to class",classImg)

            k = cv2.waitKey(1)
            if k == 27:
                continue
            #cv2.destroyAllWindows()
    else:

        test = TestConvSeq()
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
         
