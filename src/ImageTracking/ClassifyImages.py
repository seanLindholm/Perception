# This is the begining of a class which will define the model, 
# and be used for tracking cups, books and boxes on a convaier belt. 
# The model is build using Support Vector Machines (SVM), 
# since this is the method covered doing the class 313932.
# Good site for building it all https://rpubs.com/Sharon_1684/454441
import pandas as pd
import pickle as pl
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import threading
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import logging
import time
import cv2
from skimage.feature import hog



class ClassifyImages:

    def __init__(self,dataset='dataset.csv',num_of_threads=4,load_model = False):
        if load_model:
            self.__loadPretrainedModel()
        self.dataframe = pd.read_csv(dataset,sep=',',encoding='utf8')
        self.__numberOfThreads = num_of_threads

    def startAndExecuteThreadWork(self,thread_list):
        for t in thread_list:
            t.start()

        for t in thread_list:
            t.join()

    def createTrainingData(self):
        # This function should loop over each image in the dataset and convert
        # this into a readable flatten array. But to keep the feature mappings
        # consistent accros all images, we need to do some resizing and grayscaling
        # This log is just for convinience, since some pictures might not get loaded correctly
        # which doesn't have to stop the process, but can be told to the user.
        logging.basicConfig(filename="trainingDataGeneration.log", 
                            level=logging.INFO,
                            filemode='w',
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%d-%m-%Y %H:%M:%S')
        logging.info("Strating to generate training data")
        self.__error_count = self.__data_count = 0
        self.target = []
        self.dataset = []
        self.categoryname = []
        
        thread_list = []
        start = time.time()
        for path,cat,label in zip(self.dataframe["imagepath"],self.dataframe["category"],self.dataframe["label"]):
           
           thread_list.append(threading.Thread(target=self.extractInfo,args=(path,cat,label,logging,)))

           if len(thread_list) == self.__numberOfThreads:
                self.startAndExecuteThreadWork(thread_list)
                thread_list = []
           

           #self.extractInfo(path,cat,label,logging)
        self.startAndExecuteThreadWork(thread_list)

        end = time.time()
        print("Dataset created with {} thread(s) after {} seconds".format(self.__numberOfThreads,(end-start)))
        self.dataset = np.array(self.dataset)
        self.categoryDict = dict(zip(self.target,self.categoryname))
        self.target = np.array(self.target)
        logging.info("The creation of the training data ran, with {} error(s), and has generated {} number(s) of training data".format(self.__error_count,self.__data_count))
        
    
    def extractInfo(self,path,cat,label,logging):
            # Get image from the path in the dataset.csv
            img,res = self.__getimage(path)
            if res == 0:
                logging.error("The image on path {} with category {} could not be read".format(path,cat))
                self.__error_count += 1
            else:
                self.dataset.append(self.__makefeatures(img))
                self.__data_count += 1
                self.target.append(label)
                self.categoryname.append(cat)

    def classify_img(self,input_img):
        # This method takes an image as input and return the predicted class.
        # This should be used in conjunction with the object tracking to identify
        # What object is on the track.

        # The input image should be a BGR image from cv2.imread in order to function
        if type(input_img) is not np.ndarray:
            print("The input is not and numpy.ndarray, please read the image using cv2.imread," +
                  "and input that, in its original version")
            return "ImageIsOfWrongType"
        img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)

        # resize the image to 80 x 80
        img = cv2.resize(img,(80,80))
        test = self.__makefeatures(img).reshape(1,-1)
        classification = self.categoryDict[self.model.predict(test)[0]]
        return classification


    def __loadPretrainedModel(self):
        # This method load a saved model, and is ment to be used when you don't want to 
        # retrain the model. 
        self.model,self.categoryDict,self.dataset,self.target = pl.load(open("./model_save/model.p","rb"))
        pass

    def train_model(self,split=.3): 
        # Here we train the model, with our training data.
        # Next we would like to save the model, such that we don't have to retrain everytime we 
        # Need to access the model.
        X_train,X_test,y_train,y_test = train_test_split(self.dataset,self.target,test_size=split)
        self.model = sk.svm.SVC(kernel='rbf',gamma='scale',C=1)
        self.model.fit(X_train,y_train)
        y_pred = self.model.predict(X_test)
        print("The training score -rbf: {:.2f}".format(self.model.score(X_train,y_train)))
        print("The test score -rbf: {:.2f}".format(self.model.score(X_test,y_test)))

        # Save the model
        pl.dump([self.model,self.categoryDict,self.dataset,self.target], open("./model_save/model.p","wb"))
   
    def __getimage(self,path):
        # read the image using opencv
        img = cv2.imread(path)
        if(img is None):
            return 0,0
        else:
            # convert into grayscal <- (might be discussed if we want to preserve the colors or not)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # resize the image to 80 x 80
            return cv2.resize(gray,(80,80)), 1
    
    def __makefeatures(self,img):
        # This method takes the image, and flattens it. 
        # It is also possible here to append some extra feature descriptors like:
        # The histogram of oriented gradients (HOG), Edge images and many others.
        # This can improve accuracy, but also adds to the feature space/dimentionality
        features = []
        # First we flatten the image
        flatten = img.flatten()
        # Some other features
        # Right now there is no extra feautres, but so far it seems like the
        # biggest problem is distinguishing boxes from book.
        # One feature idea might be to add edge detection, to try and find 
        # text on the front page of the books.
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
        more_features = hog_image.flatten()
        # horizontal stack them
        features = np.hstack((flatten,more_features))

        # Use pca to reduce dimentionality
        # TODO make this, i think it could be nice

        return features

# Debugging 
if __name__ == "__main__":
    # This is for testing the training aspect
    t = ClassifyImages()
    t.createTrainingData()
    print(t.dataset.shape)
    print(t.target.shape)
    t.train_model()

    #img = cv2.imread("./images/book_00004.jpg")
    #print("classification: ",t.classify_img(img))

    
    # This can be used to load old model
    #t2 = ClassifyImages(load_model=True)

    # This is for testing the model
    image = cv2.imread("./images/box_00024.jpg")
    #print("classification: ", t2.classify_img(img))