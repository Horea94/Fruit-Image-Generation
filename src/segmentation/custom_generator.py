import math

import numpy as np
import os
from PIL import Image
from keras.utils import Sequence

import util


class CustomDataGenerator(Sequence):

    def __init__(self, images_path, masks_path, batch_size=5, image_dimensions=(128, 128, 3), shuffle=False, augment=False):
        self.masks_paths = self.build_image_paths(masks_path)  # array of mask paths
        self.images_paths = self.build_image_paths(images_path)  # array of mask paths
        self.indexes = np.arange(len(self.images_paths))
        self.dim = image_dimensions  # image dimensions
        self.batch_size = batch_size  # batch size
        self.shuffle = shuffle  # shuffle bool
        self.augment = augment  # augment data bool
        self.on_epoch_end()

    def __len__(self):
        """Denotes the smallest number of batches per epoch to include all images in the train set at least once"""
        return int(math.ceil(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        # select data and load images
        masks = np.array([util.create_target_from_mask(np.array(Image.open(self.masks_paths[k]).resize(self.dim[:-1]))) for k in indexes])
        images = np.array([np.array(Image.open(self.images_paths[k]).resize(self.dim[:-1])) for k in indexes])
        return images, masks

    def build_image_paths(self, starting_path):
        image_paths = [starting_path + x for x in os.listdir(starting_path)]
        return image_paths
