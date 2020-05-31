import os
import cv2
import numpy as np
import sys

import tensorflow as tf

##############################################
learning_rate = 0.1  # initial learning rate
min_learning_rate = 0.00001  # once the learning rate reaches this value, do not decrease it further
learning_rate_reduction_factor = 0.5  # the factor used when reducing the learning rate -> learning_rate *= learning_rate_reduction_factor
patience = 3  # how many epochs to wait before reducing the learning rate when the loss plateaus
verbose = 1  # controls the amount of logging done during training and testing: 0 - none, 1 - reports metrics after each batch, 2 - reports metrics after each epoch
image_size = (100, 100)  # width and height of the used images
input_shape = (100, 100, 3)  # the expected input shape for the trained models; since the images in the Fruit-360 are 100 x 100 RGB images, this is the required input shape

use_label_file = False  # set this to true if you want load the label names from a file; uses the label_file defined below; the file should contain the names of the used labels, each label on a separate line
label_file = 'labels.txt'
base_dir = '../..'  # relative path to the Fruit-Images-Dataset folder
test_dir = os.path.join(base_dir, 'Test')
train_dir = os.path.join(base_dir, 'Training')
output_dir = 'output_files'  # root folder in which to save the the output files; the files will be under output_files/model_name
##############################################

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# if we want to train the network for a subset of the fruit classes instead of all, we can set the use_label_file to true and place in the label_file the classes we want to train for, one per line
if use_label_file:
    with open(label_file, "r") as f:
        labels = [x.strip() for x in f.readlines()]
else:
    labels = os.listdir(train_dir)
num_classes = len(labels)


# Create a custom layer that converts the original image from
# RGB to HSV and grayscale and concatenates the results
# forming in input of size 100 x 100 x 4
def convert_to_hsv_and_grayscale(x):
    hsv = tf.image.rgb_to_hsv(x)
    gray = tf.image.rgb_to_grayscale(x)
    rez = tf.concat([hsv, gray], axis=-1)
    return rez


def predict_one_file(path_to_file, name=""):
    model_out_dir = os.path.join(output_dir, name)
    if not os.path.exists(model_out_dir):
        print("Model weights not found")
        sys.exit(0)
    if not os.path.exists(path_to_file):
        print("Specified file not found")
        sys.exit(0)
    model = tf.keras.models.load_model(model_out_dir + "/model.h5")
    image = cv2.imread(path_to_file)
    image = cv2.resize(image, (100, 100))
    # data = np.ndarray(shape=(1, 100, 100, 3), dtype=np.int)
    image_array = np.asarray(image)
    image_array = np.reshape(image_array, (1,) + np.shape(image_array))
    # data[0] = image_array
    y_pred = model.predict(image_array, 1)
    print(y_pred)
    print(y_pred.argmax(axis=-1))
    print(labels[y_pred.argmax(axis=-1)[0]])


# path to the file
predict_one_file(path_to_file='../../Training/Banana/0_100.jpg', name='fruit-360 model')
