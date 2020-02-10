import os

dataset_train_folder = '../../Training/'
dataset_test_folder = '../../Test/'

background_folder = '../../Images/Backgrounds/'
train_folder = '../../Images/Train/'
image_folder = train_folder + 'images/'
mask_folder = train_folder + 'masks/'
annotation_folder = train_folder + 'annotations/'
test_folder = '../../Images/Test/'
output_folder = '../../Images/Rez/'
models_folder = 'models/'
unet_weights = models_folder + 'unet.h5'

labels_file = 'labels.txt'
with open(labels_file, mode='r') as f:
    fruit_labels = [x.strip() for x in f.readlines()]
fruit_labels.sort()
fruit_labels = ['Background'] + fruit_labels

batch_size = 5
epochs = 20
img_size = (256, 256, 3)  # height, width, channels

####################################################################
# used for segmentation only; i.e. we want to find which pixels belong to a fruit without classifying it, the classification should be done with a separate model
# the model will classify the pixels in background(black) and fruit(white)
is_binary_classification_task = True

if is_binary_classification_task:
    num_classes = 2  # used in conjunction with the colormap for segmentation and classification
    color_map = {0: (0, 0, 0),
                 1: (255, 255, 255)}
else:
    num_classes = len(fruit_labels)
    color_map = {0: (0, 0, 0),
                 1: (255, 0, 0),
                 2: (0, 255, 0),
                 3: (0, 0, 255),
                 4: (128, 0, 128)}

####################################################################

# min/max width and height of images that are used to build the training data for each class
min_fruit_size = 20
max_fruit_size = 150

mask_threshold = 246  # threshold used for generating masks
# number of images to generate in the segmentation dataset
# for each generated image, the corresponding mask is also generated
# so the total number of generated images is 2 * dataset_generation_limit
dataset_generation_limit = 5
# number of threads that build the dataset
# the load is balanced among the threads
total_threads = 1

####################################################################
