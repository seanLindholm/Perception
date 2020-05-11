# This is the begining of a class which will define the model, 
# and be used for tracking cups, books and boxes on a convaier belt. 
# The model is build using Support Vector Machines (SVM), 
# since this is the method covered doing the class 313932.
# Good site for building it all https://rpubs.com/Sharon_1684/454441
import pandas as pd
import pickle as pl
import numpy as np
import sklearn as sk
import imutils
import sys
import matplotlib.pyplot as plt
import threading
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import logging
import time
import cv2
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


resize_ = 200
class ClassifyImages:

    def __init__(self,dataset='dataset.csv',num_of_threads=1,load_model=False,load_path="./model_save"):
        if load_model:
            self.__loadPretrainedModel(load_path)
        else:
            self.dataframe = pd.read_csv(dataset,sep=',',encoding='utf8')
            self.negDataframe = pd.read_csv('dataset_negativ.csv',sep=',',encoding='utf8')
            self.__numberOfThreads = num_of_threads
        
        self.categoryDict = {
            0:"cup",
            1:"book",
            2:"box",
            3:"Nothing"
            }

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
        self.negDataset = []
        self.negTarget = []
        self.categoryname = []
        
        thread_list = []
        start = time.time()
        # Get all the positiv data
        for path,cat,label in zip(self.dataframe["imagepath"],self.dataframe["category"],self.dataframe["label"]):
           
           thread_list.append(threading.Thread(target=self.extractInfo,args=(path,cat,label,logging,False,)))

           if len(thread_list) == self.__numberOfThreads:
                self.startAndExecuteThreadWork(thread_list)
                thread_list = []


        self.startAndExecuteThreadWork(thread_list)

        #Get all the negativ data
        for path,cat,label in zip(self.negDataframe["imagepath"],self.negDataframe["category"],self.negDataframe["label"]):
        
            thread_list.append(threading.Thread(target=self.extractInfo,args=(path,cat,label,logging,True,)))

            if len(thread_list) == self.__numberOfThreads:
                self.startAndExecuteThreadWork(thread_list)
                thread_list = []
        
        end = time.time()
        print("Dataset created with {} thread(s) after {} seconds".format(self.__numberOfThreads,(end-start)))
        self.dataset = np.array(self.dataset)
        self.target = np.array(self.target)

        self.negDataset = np.array(self.negDataset)
        self.negTarget = np.array(self.negTarget)

       
        logging.info("The creation of the training data ran, with {} error(s), and has generated {} number(s) of training data".format(self.__error_count,self.__data_count))
        
    
    def extractInfo(self,path,cat,label,logging,negData):
            # Get image from the path in the dataset.csv
            img,res = self.__getimage(path)
            if res == 0:
                logging.error("The image on path {} with category {} could not be read".format(path,cat))
                self.__error_count += 1
            else:
                if not negData:
                    self.dataset.append(self.__makefeatures(img))
                    self.target.append(label)
                    self.__data_count += 1
                    self.categoryname.append(cat)
    
                else:
                    self.negDataset.append(self.__makefeatures(img))
                    self.negTarget.append(label)

    def __getimage(self,path):
        # read the image using opencv
        img = cv2.imread(path)
        if img is None:
            return 0,0
        else:
            gray = img
        
            #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # resize the image to resize variable
            return cv2.resize(gray,(resize_,resize_)), 1
    
    def __makefeatures(self,img):
        # This method takes the image, and flattens it. 
        # It is also possible here to append some extra feature descriptors like:
        # The histogram of oriented gradients (HOG), Edge images and many others.
        # This can improve accuracy, but also adds to the feature space/dimentionality
        # Some other features
        # biggest problem is distinguishing boxes from book.
        fd = hog(img, orientations=9, pixels_per_cell=(16,16),
                            cells_per_block=(4,4),block_norm='L2', visualize=False, 
                            transform_sqrt=True, feature_vector=True, multichannel=True)        
        
        more_features = fd
        
        #Calculating color histograms of the picture using 16 bins. 
        numPixels = np.prod(img.shape[:2])
        (b, g, r) = cv2.split(img)
        histogramR = cv2.calcHist([r], [0], None, [16], [0, 255]) / numPixels
        histogramG = cv2.calcHist([g], [0], None, [16], [0, 255]) / numPixels
        histogramB = cv2.calcHist([b], [0], None, [16], [0, 255]) / numPixels
        # horizontal stack them
        features = np.hstack((more_features,histogramR.flatten()))
        features = np.hstack((features,histogramG.flatten()))
        features = np.hstack((features,histogramB.flatten()))
        return features
 
    def train_model(self,split=.3): 
        # Here we train the model, with our training data.
        # Next we would like to save the model, such that we don't have to retrain everytime we 
        # Need to access the model.
        X_train,X_test,y_train,y_test = train_test_split(self.dataset,self.target,test_size=split,shuffle=True)
    
        X_train = np.vstack((X_train,self.negDataset))
        y_train = np.hstack((y_train,self.negTarget))

        self.model = sk.svm.SVC(kernel='linear',C=1,probability=True)
        self.model.fit(X_train,y_train)
        yfit = self.model.predict(X_test)

        print("The training score -rbf: {:.2f}".format(self.model.score(X_train,y_train)))
        print("The test score -rbf: {:.2f}".format(self.model.score(X_test,y_test)))

        #Save the model
        self.__saveModels()
       
    def classify_img(self,input_img,GaryTheImage=True):
        # This method takes an image as input and return the predicted class.
        # This should be used in conjunction with the object tracking to identify
        # What object is on the track.

        if type(input_img) is not np.ndarray:
            print("The input is not and numpy.ndarray, please read the image using cv2.imread," +
                  "and input that, in its original version")
            return "ImageIsOfWrongType"

        if GaryTheImage:
            input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
        # resize the image to resize variable
        img = cv2.resize(input_img,(resize_,resize_))
        test = self.__makefeatures(img).reshape(1,-1)
        prob1 = self.model.predict_proba(test)[0]
        classification = self.categoryDict[np.argmax(prob1)]
        sys.stdout.flush()
        return classification,prob1


    def __loadPretrainedModel(self,load_path):
        # This method load a saved model, and is ment to be used when you don't want to 
        # retrain the model. 
        self.model = pl.load(open(load_path + "/model.p","rb"))[0]
        self.dataset = pl.load(open(load_path + "/dataset.p","rb"))[0]
        self.target = pl.load(open(load_path + "/target.p","rb"))[0]
        self.negDataset = pl.load(open(load_path + "/negativDataSet.p","rb"))[0]
        self.negTarget = pl.load(open(load_path + "/negativTarget.p","rb"))[0]


    def __saveModels(self):
        # This method load a saved model, and is ment to be used when you don't want to 
        # retrain the model. 
        pl.dump([self.model],open("./model_save/model.p","wb"))
        pl.dump([self.dataset],open("./model_save/dataset.p","wb"))
        pl.dump([self.target],open("./model_save/target.p","wb"))
        pl.dump([self.negDataset],open("./model_save/negativDataSet.p","wb"))
        pl.dump([self.negTarget],open("./model_save/negativTarget.p","wb"))


if __name__ == "__main__":
    # This is for testing the training aspect
    
    t = ClassifyImages()
    
    print("Creating training data with images being {0}x{0}. This can take some time".format(resize_))
    sys.stdout.flush()
    t.createTrainingData()

    print("Begining trainig of model")
    sys.stdout.flush()

    t.train_model()

    # This is for testing the model
    img = cv2.imread("./images/book_01004.jpg")
    print("classification: ", t.classify_img(img,False))

    # This is for testing the model
    img = cv2.imread("./images/box_01055.jpg")
    print("classification: ", t.classify_img(img,False))

    # This is for testing the model
    img = cv2.imread("./images/cup_01089.jpg")
    print("classification: ", t.classify_img(img,False))
