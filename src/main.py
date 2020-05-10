from ImageTracking.ClassifyImages import ClassifyImages
# import the calibration as well
import cv2
import numpy as np
import os
import time
import sys 

background = cv2.imread("/Users/oliverbehrens/Desktop/left/1585434282_261431932_Left.png")
cv2.imshow("background", background)
#background = cv2.imread("ImageTracking/Stereo_conveyor_with_occlusions/left/1585434751_931764603_Left.png")



def runVideoStreamNoOcculison():
    # Get the number of images in the left folder (The Video)
    # There is the same number of images in both left and right folder.
    path_left = "/Users/oliverbehrens/Desktop/left"
    path_right = "/Users/oliverbehrens/Desktop/right"
    
    left_pics = sorted([f for f in os.listdir(path_left)])
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
    path_left = "ImageTracking/Stereo_conveyor_with_occlusions/left"
    path_right = "ImageTracking/Stereo_conveyor_with_occlusions/right"
    
    left_pics = [f for f in os.listdir(path_left)]
    right_pics = [f for f in os.listdir(path_right)]
    left_pics = sorted(left_pics)
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
    
    #img = cv2.imread("./images/book_00004.jpg")
    #print("classification: ",t.classify_img(img))

    if not test:
        
        
        video = runVideoStreamNoOcculison()

        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        frame = 0

        Count = np.zeros(3)
        for left,right in video:
            cl = "0"
            mean = []
            frame += 1
            mask_object,boarder,mean = seqmentObject(left)
            #class_ = model.classify_img(mask_object,True)
            #contours
            #left = cv2.bitwise_and(left,left,mask=mask_object)
            print(mean)
            if boarder != []:
                c_img = left[mean[0]-120:mean[0]+120,mean[1]-120:mean[1]+120,:]
                cv2.rectangle(left, boarder[0][0], boarder[0][1], (0, 255, 0), 2)
                if mean[1]<1050:
                    #c_img = clone[y:y+winH, x:x+winW,:]
                    cv2.imshow("Crop",c_img)
                    cl = str(model.classify_img(c_img,False))
                    cll = model.classify_img(c_img,False)
                    if cll[0] == "cup":
                        Count[0] = Count[0]+cll[1]*1
                    elif cll[0] == "box":
                        Count[2] = Count[2]+ cll[1]*1
                    elif cll[0] =="book":
                        Count[1] = Count[1] + cll[1]*1
                    cl = model.categoryDict[np.argmax(Count)]
                elif mean[1]>1050: 
                    Count = np.zeros(3)
            print(Count)
                

                    
                    


            cv2.putText(left,cl + " - " + str(frame), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
            cv2.imshow("left",left)
            #cv2.imshow("image to class",classImg)
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
         
