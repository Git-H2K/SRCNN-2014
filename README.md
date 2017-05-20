# SRCNN-2014
Paper : http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf

# Coding Environment
Python 3.52

Tensorflow

Opencv 3.2.0

# Files
Main_training : Train CNN network

Main_test : Test your bmp file

GENERATE_TRAINING_DATA : Read training bmp files and generate training patches

GENERATE_TESTING_DATA : Read a test bmp file

NETWORK : First layer-[9,9,1,64] / Second layer-[1,1,63,32] / Thir layer-[5,5,32,1]
