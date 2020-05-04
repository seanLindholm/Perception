# This is the begining of a class which will define the model, 
# and be used for tracking cups, books and boxes on a convaier belt. 
# The model is build using Support Vector Machines (SVM), 
# since this is the method covered doing the class 313932.
# Good site for building it all https://rpubs.com/Sharon_1684/454441
import pandas as pd
import pickle as pl
import numpy as np
import sklearn as sk
import sys
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
from sklearn.decomposition import PCA



resize_ = 200
class ClassifyImages:

    def __init__(self,dataset='dataset.csv',num_of_threads=4,load_model=False,load_path="./model_save"):
        if load_model:
            self.__loadPretrainedModel(load_path)
        else:
            self.dataframe = pd.read_csv(dataset,sep=',',encoding='utf8')
            self.negDataframe = pd.read_csv('dataset_negativ.csv',sep=',',encoding='utf8')
            self.__numberOfThreads = num_of_threads
            self.pca = PCA(.98)
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
                else:
                    self.negDataset.append(self.__makefeatures(img))
                    self.negTarget.append(label)

                self.__data_count += 1
                self.categoryname.append(cat)

    def __getimage(self,path):
        # read the image using opencv
        img = cv2.imread(path)
        if img is None:
            return 0,0
        else:
            # convert into grayscal <- (might be discussed if we want to preserve the colors or not)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # resize the image to resize variable
            return cv2.resize(gray,(resize_,resize_)), 1
    
    def __makefeatures(self,img):
        # This method takes the image, and flattens it. 
        # It is also possible here to append some extra feature descriptors like:
        # The histogram of oriented gradients (HOG), Edge images and many others.
        # This can improve accuracy, but also adds to the feature space/dimentionality
        # Some other features
        # Right now there is no extra feautres, but so far it seems like the
        # biggest problem is distinguishing boxes from book.
        # One feature idea might be to add edge detection, to try and find 
        # text on the front page of the books.
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=False)
        more_features = fd.flatten()
        # horizontal stack them
        features = np.hstack(([],more_features))

        return features
 
    def train_model(self,split=.3): 
        # Here we train the model, with our training data.
        # Next we would like to save the model, such that we don't have to retrain everytime we 
        # Need to access the model.
        X_train,X_test,y_train,y_test = train_test_split(self.dataset,self.target,test_size=split,shuffle=True)
        print(X_train.shape)
        print(y_train.shape)
        print()
        X_train = np.vstack((X_train,self.negDataset))
        y_train = np.hstack((y_train,self.negTarget))
        print(X_train.shape)
        print(y_train.shape)
        print()

        self.std = StandardScaler().fit(X_train)
        X_train = self.std.transform(X_train)
        X_test = self.std.transform(X_test)
        
        #PCA fitting and transforming
        self.pca = self.pca.fit(X_train)
        X_train = self.pca.transform(X_train)
        X_test = self.pca.transform(X_test)

        self.model = sk.svm.SVC(kernel='linear',C=1,probability=True)
        self.model.fit(X_train,y_train)
        print("The training score -rbf: {:.2f}".format(self.model.score(X_train,y_train)))
        print("The test score -rbf: {:.2f}".format(self.model.score(X_test,y_test)))

        #Save the model
        self.__saveModels()
       
    def classify_img(self,input_img,GaryTheImage=True):
        # This method takes an image as input and return the predicted class.
        # This should be used in conjunction with the object tracking to identify
        # What object is on the track.

        # The input image should be a BGR image from cv2.imread in order to function
        if type(input_img) is not np.ndarray:
            print("The input is not and numpy.ndarray, please read the image using cv2.imread," +
                  "and input that, in its original version")
            return "ImageIsOfWrongType"
        if GaryTheImage:
            input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
        # resize the image to resize variable
        img = cv2.resize(input_img,(resize_,resize_))
        test = self.__makefeatures(img).reshape(1,-1)
        hogs = test.copy()
        test = self.std.transform(test)
        test = self.pca.transform(test)
        prob = self.model.predict_proba(test)[0]
        classification = self.categoryDict[np.argmax(prob)]
        sys.stdout.flush()
        return classification,prob[np.argmax(prob)],hogs


    def __loadPretrainedModel(self,load_path):
        # This method load a saved model, and is ment to be used when you don't want to 
        # retrain the model. 
        self.model = pl.load(open(load_path + "/model.p","rb"))[0]
        self.pca = pl.load(open(load_path + "/pca.p","rb"))[0]
        self.std = pl.load(open(load_path +"/standardscaler.p","rb"))[0]
        self.dataset = pl.load(open(load_path + "/dataset.p","rb"))[0]
        self.target = pl.load(open(load_path + "/target.p","rb"))[0]
        self.negDataset = pl.load(open(load_path + "/negativDataSet.p","rb"))[0]
        self.negTarget = pl.load(open(load_path + "/negativTarget.p","rb"))[0]


    def __saveModels(self):
        # This method load a saved model, and is ment to be used when you don't want to 
        # retrain the model. 
        pl.dump([self.model],open("./model_save/model.p","wb"))
        pl.dump([self.pca],open("./model_save/pca.p","wb"))
        pl.dump([self.std],open("./model_save/standardscaler.p","wb"))
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
    print("Done traning, starting hard negative training")
    sys.stdout.flush()

    #after first training, make hard_negative from all images in negative set:
    for path,cat,label in zip(t.negDataframe["imagepath"],t.negDataframe["category"],t.negDataframe["label"]):
           
        image = cv2.imread(path)
        boarder,hard_negative = findObjectWithSliding(image,t,True)
        
        if hard_negative != []:
            hard_negative = np.squeeze(np.array(hard_negative),axis=1)
            negative_label = np.array([ 3 for _ in range(hard_negative.shape[0])])
            t.negDataset = np.vstack((t.negDataset,hard_negative))
            t.negTarget = np.hstack((t.negTarget,negative_label))

    print("Located hard negatives, retraining model")
    sys.stdout.flush()
    #Traing the model again, with the new set.
    t.train_model()
    
    print("Testing on one image")
    sys.stdout.flush()
    image_cup = cv2.imread("C:\\Users\\swang\\Desktop\\Video\\NoOcclusions\\left\\1585434318_469291925_Left.png")
    boarder,hard_negative = findObjectWithSliding(image_cup,t,False)
    if boarder != []:    
        for xy,xWinyWin,_ in boarder:
            cv2.rectangle(image_cup, xy, xWinyWin, (0, 255, 0), 2)
   
    cv2.imshow("The cup",image_cup)
    cv2.waitKey(20000)
    
    #img = cv2.imread("./images/book_00004.jpg")
    #print("classification: ",t.classify_img(img))
    
    
    # This can be used to load old model
    t2 = ClassifyImages(load_model=True)
    image_cup = cv2.imread("C:\\Users\\swang\\Desktop\\Video\\NoOcclusions\\left\\1585434327_197681904_Left.png")
    boarder,hard_negative = findObjectWithSliding(image_cup,t2,False)
    print(boarder)
    if boarder != []:    
        for xy,xWinyWin,_ in boarder:
            cv2.rectangle(image_cup, xy, xWinyWin, (0, 255, 0), 2)
   
    cv2.imshow("The cup",image_cup)
    cv2.waitKey(5000)
    # This is for testing the model
    img = cv2.imread("./images/book_01004.jpg")
    print("classification: ", t2.classify_img(img,True))

    # This is for testing the model
    img = cv2.imread("./images/box_01055.jpg")
    print("classification: ", t2.classify_img(img,True))

    # This is for testing the model
    img = cv2.imread("./images/cup_01090.jpg")
    print("classification: ", t2.classify_img(img,True))
