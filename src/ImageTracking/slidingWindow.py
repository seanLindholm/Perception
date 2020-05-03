import imutils
import cv2
from ClassifyImages import ClassifyImages
import time
import sys
import numpy as np

def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image
        
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(250, image.shape[0], stepSize):
        if y > 500:
            break
        for x in range(350, image.shape[1], stepSize):
            # yield the current window
            if x > 950:
                break
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
        

winW = 200
winH = 200 
min_prob = 0
def readProb(elem):
    return elem[1]

def findObjectWithSliding(image,model,hard_negativ):

    
    boarderCup = []
    boarderBook = []
    boarderBox = []
    boarder = []
    neg_boarder = []
    Box_count = Cup_count = Book_count = 0
    class_ = []
    prob_list = []

    for resized in pyramid(image, scale=2):
        #cv2.line(resized, (0, 100), (resized.shape[1], 100), (255, 0, 0), 1, 1)
        #cv2.line(resized, (0, 200), (resized.shape[1], 200), (255, 0, 0), 1, 1)
        #cv2.line(resized, (0, 300), (resized.shape[1], 300), (255, 0, 0), 1, 1)
        #cv2.line(resized, (0, 400), (resized.shape[1], 400), (255, 0, 0), 1, 1)
        #cv2.line(resized, (100, 0), (100, resized.shape[0]), (255, 0, 0), 1, 1)
        #cv2.line(resized, (200, 0), (200,resized.shape[0]), (255, 0, 0), 1, 1)
        #cv2.line(resized, (300, 0), (300, resized.shape[0]), (255, 0, 0), 1, 1)
        #cv2.line(resized, (400, 0), (400, resized.shape[0]), (255, 0, 0), 1, 1)
        #cv2.line(resized, (500, 0), (500, resized.shape[0]), (255, 0, 0), 1, 1)
        #cv2.line(resized, (600, 0), (600, resized.shape[0]), (255, 0, 0), 1, 1)
        #cv2.line(resized, (700, 0), (700, resized.shape[0]), (255, 0, 0), 1, 1)
        #cv2.line(resized, (800, 0), (800, resized.shape[0]), (255, 0, 0), 1, 1)

        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=30, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            clone = resized.copy()
            c_img = clone[y:y+winH, x:x+winW,:]
            Class,prob,hog = model.classify_img(c_img)
            if not hard_negativ:
                copy = image.copy()
                cv2.rectangle(copy, (x,y), (x+winW,y+winH), (0, 255, 0), 2)
                cv2.imshow("Window", copy)
                cv2.waitKey(1)
            
            sys.stdout.flush()
            if Class != "Nothing":
                #save boarder and procent
                
                if prob > min_prob:
                    if Class == "cup":
                        boarderCup.append([(x, y), (x + winW, y + winH),prob])
                        Cup_count += 1
                    
                    elif Class == "book":
                        boarderBook.append([(x, y), (x + winW, y + winH),prob])
                        Book_count += 1
                
                    elif Class == "box":
                        boarderBox.append([(x, y), (x + winW, y + winH),prob])
                        Box_count += 1
                    
                if hard_negativ:
                    #Mark all of them as noting, and save then as part of new traning set.
                    neg_boarder.append(hog)
                    prob_list.append(prob)
                    
        boarder.append(boarderCup)
        boarder.append(boarderBook)
        boarder.append(boarderBox)
        cnt = np.array([Cup_count,Book_count,Box_count])  
        
        if hard_negativ:
                #Mark all of them as noting, and save then as part of new traning set.
                neg = [hog for _,hog in sorted(zip(prob_list,neg_boarder),reverse=True)]
                neg_boarder = neg
        else:
            max_ = np.argmax(cnt)
            if max_ == 0:
                print("Cup dominant")
            elif max_ == 1:
                print("Book dominant")
            elif max_ == 2:
                print("Box dominant")
            boarder = boarder[np.argmax(cnt)]

        return boarder,neg_boarder
