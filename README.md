# Results and how to run the code
The results can be seen under src/Video, where the original capture video has been speed up to around 24 frames per second, for your viewing pleasure. The solution is not a realtime solution, and only runs with around 4 frames per second.

In order to run the code, you will need to navigate into *src/constants.py*. 
Here you need to specify the paths to the stream of images on your local machine. 
You also have the option here to change between viewing the video with occlusion and the video without by changing the boolean there.

When you are all set, you can then run the main script by typing:

python main.py

# Perception
This repo holds the answer to the final assignment in the course 31392 (Perception for autonomous systems)

# Data
The folder Images inside src/ImageTracking/images holds all the training images, most of them are scrapped from google but some are also taken by handheld devices.
The images which is used as input are all handled inside the script (ClassifyImage) where it reads it using cv2.imread and resize it down to an 200x200.
The data also get some negatives in (which is mainly pictures of the empty convayor setting), this reduces the confussion a bit, when trying to classify the objects from the conveyor.

# Add data
If you wish to add data then you will need to add a line to *dataset.csv* inside src/ImageTracking, here you will need to specify the path to the image, and then the name category alongside a numerical label.

# The model
The model is a Support Vector Machine (https://en.wikipedia.org/wiki/Support-vector_machine), used from the sklearn python library.

# Training on data
If you wish to retrain you would need to navigate to src/ImageTracking. Here you can run the command

python ClassifyImages.py 

Which will retrain and save a new model.
