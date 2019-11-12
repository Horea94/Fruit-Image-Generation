import os

dataset_train_folder = '../../Training/'
dataset_test_folder = '../../Test/'

background_folder = '../../Segmentation/Backgrounds/'
train_folder = '../../Segmentation/Train/'
image_folder = train_folder + 'images/'
mask_folder = train_folder + 'masks/'
test_folder = '../../Segmentation/Test/'
output_folder = '../../Segmentation/Rez/'
models_folder = 'models/'
unet_weights = models_folder + 'unet.h5'

labels_file = 'labels.txt'
with open(labels_file, mode='r') as f:
    labels = [x.strip() for x in f.readlines()]
labels.sort()
labels = ['Background'] + labels

num_classes = len(labels)

img_size = (256, 256, 3)  # height, width, channels

color_map = {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0),
             3: (0, 0, 255),
             4: (128, 0, 128)}


# min/max width and height of images that are used to build the training data for each class
min_fruit_size = 25
max_fruit_size = 80

mask_threshold = 246  # threshold used for generating masks
# number of images to generate in the segmentation dataset
# for each generated image, the corresponding mask is also generated
# so the total number of generated images is 2 * dataset_generation_limit
dataset_generation_limit = 5000
# number of threads that build the dataset
# the load is balanced among the threads
total_threads = 5

batch_size = 5
epochs = 5
