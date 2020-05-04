from ImageTracking.ClassifyImages import ClassifyImages
# import the calibration as well
import cv2
import numpy as np
import os
import time
import sys 

background = cv2.imread("C:\\Users\\swang\\Desktop\\Video\\NoOcclusions\\left\\1585434282_261431932_Left.png")



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

def seqmentObject(img):
    image = img.copy()
    #Complete removal of backgroubd
    image = background-image
    image[image>200] = 0
    image[image<50] = 0
    erode_ = np.ones((6,6),np.uint8)
    dialte_ = np.ones((10,10),np.uint8)
    image = cv2.erode(image,erode_)
    image = cv2.dilate(image,dialte_)
    boarder = []
    mean = []
    #locate the object in the image
    seq = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    seq = seq[250:,350:1200]
    xy = np.where(seq> 10)
    x = xy[0].mean() if xy[0].size > 0 else np.nan     
    y = xy[1].mean() if xy[1].size > 0 else np.nan      
    try:         
        x = int(x)+250
        y = int(y)+350
        mean = [x,y]
        boarder.append([(y-160,x+160), (y+160,x-160)])
    except Exception:         
        pass
    
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),boarder,mean

if __name__ == "__main__":
    test = False
    model = ClassifyImages(load_model=True,dataset = "./ImageTracking/dataset.csv", load_path="./ImageTracking/model_save")
        
    if not test:
        
        
        video = runVideoStreamNoOcculison()

        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,50)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        frame = 0
        
        for left,right in video:
            frame += 1
            mask_object,boarder,mean = seqmentObject(left)
            #class_ = model.classify_img(mask_object,True)
            #contours
            left = cv2.bitwise_and(left,left,mask=mask_object)
            if boarder != []:
                c_img = left[mean[0]-80:mean[0]+80,mean[1]-80:mean[1]+80,:]
                cv2.rectangle(left, boarder[0][0], boarder[0][1], (0, 255, 0), 2)
                class_,prob,_ = model.classify_img(c_img,False)              
                cv2.putText(left,class_ + " - " + str(frame) + " prob: " + str(prob), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
                cv2.imshow("Crop",c_img)
            cv2.imshow("left",left)
           # cv2.imshow("left",left)
           # cv2.imshow("image to class",classImg)
            cv2.waitKey(1)
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
         
