# Perception
This repo holds the answer to the final assignment in the course 31392 (Perception for autonomous systems)

# Data
The folder Images inside src->ImageTracking holds all the training images, some are scrapped from google but most of them are taken by handheld devises in everyday setting.

# Add data
If you wish to add data then you will need to add a line to *dataset.csv* inside src->ImageTracking, here you will need to specify the path to the image, and then the name category alongside a numerical label.

# The model
The model is a Support Vector Machine (https://en.wikipedia.org/wiki/Support-vector_machine), used from the sklearn python library.
