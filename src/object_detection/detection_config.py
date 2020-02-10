import numpy as np
import math

dataset_train_folder = '../../Training/'
dataset_test_folder = '../../Test/'

background_folder = '../../Images/Backgrounds/'
train_folder = '../../Images/Train/'
image_folder = train_folder + 'images/'
mask_folder = train_folder + 'masks/'
annotation_folder = train_folder + 'annotations/'
test_folder = '../../Images/Test/Input'
output_folder = '../../Images/Rez/'
models_folder = 'models/'

labels_file = 'labels.txt'
with open(labels_file, mode='r') as f:
    fruit_labels = [x.strip() for x in f.readlines()]
fruit_labels.sort()
bg = 'Background'
fruit_labels = fruit_labels + [bg]
num_classes = len(fruit_labels)
class_to_color = {fruit_labels[v]: np.random.randint(0, 255, 3) for v in range(num_classes)}

####################################################################
# batch_size = 5
epochs = 100
input_shape_img = (None, None, 3)  # height, width, channels
img_size = (256, 256, 3)  # height, width, channels

color_map = {0: (0, 0, 0),
             1: (255, 255, 255)}

# data augmentation
use_horizontal_flips = False
use_vertical_flips = False
random_rotate = False

# balanced_classes = True

# anchor box scales
# anchor_box_scales = [128]
anchor_box_scales = [64, 128]
# anchor box ratios
# anchor_box_ratios = [[1, 1]]
anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
# size to resize the smallest side of the image
im_size = 256
# number of ROIs at once
num_rois = 10
# stride at the RPN (this depends on the network configuration)
rpn_stride = 8

# img_channel_mean = [103.939, 116.779, 123.68]
# img_scaling_factor = 1.0
std_scaling = 4.0
classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

# overlaps for RPN
rpn_min_overlap = 0.3
rpn_max_overlap = 0.7

# overlaps for classifier ROIs
classifier_min_overlap = 0.3
classifier_max_overlap = 0.7

# learning rates for rpn and classifier
initial_rpn_lr = 0.1
min_rpn_lr = 1e-10
initial_cls_lr = 0.1
min_cls_lr = 1e-10

####################################################################

# min/max width and height of images that are used to build the training data for each class
min_fruit_size = 64
max_fruit_size = 128

mask_threshold = 246  # threshold used for generating masks
# number of images to generate in the segmentation dataset
# for each generated image, the corresponding mask is also generated
# so the total number of generated images is 2 * dataset_generation_limit
dataset_generation_limit = 300
# number of threads that build the dataset
# the load is balanced among the threads
total_threads = 1

####################################################################
