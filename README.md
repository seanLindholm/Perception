# Perception
This repo holds the answer to the final assignment in the course 31392 (Perception for autonomous systems)

# Data
The folder Images inside src->ImageTracking holds all the training images, some are scrapped from google but most of them are taken by handheld devices in everyday setting.
The images which is used as input are all handled inside the script (ClassifyImage) where it reads it using cv2.imread and resize it down to an 80x80 grayscale image.


# Add data
If you wish to add data then you will need to add a line to *dataset.csv* inside src->ImageTracking, here you will need to specify the path to the image, and then the name category alongside a numerical label.

# The model
The model is a Support Vector Machine (https://en.wikipedia.org/wiki/Support-vector_machine), used from the sklearn python library.

# Training on data
If you wish to retrain the model a .log file is created after training, in here you can see if some of the traininig data could not be read, and how many training data was created.
