import math
import numpy as np
import os
from keras.utils import Sequence
from PIL import Image


class CustomDataGenerator(Sequence):

    def __init__(self, annotations_path, images_path, batch_size=1, image_dimensions=(None, None, 3), shuffle=False, augment=False):
        self.image_data = self.build_image_data(annotations_path, images_path)  # array of image data
        self.indexes = np.arange(len(self.image_data))
        self.dim = image_dimensions  # image dimensions
        self.batch_size = batch_size  # batch size
        self.shuffle = shuffle  # shuffle bool
        self.augment = augment  # augment data bool
        self.on_epoch_end()

    def __len__(self):
        """Denotes the smallest number of batches per epoch to include all images in the train set at least once"""
        return int(math.ceil(len(self.image_data) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.image_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        # select data and load images
        images_batch = np.array([np.array(Image.open(self.image_data[k]['filepath'])) for k in indexes])
        bboxes = [self.image_data[k]['bboxes'] for k in indexes]

    def build_image_data(self, annotations_path, images_path):
        data_map = {}
        rez = []
        for annotation in os.listdir(annotations_path):
            annotation_file = annotations_path + annotation
            with open(annotation_file, 'r') as f:
                lines = [x.strip() for x in f.readlines()]
                image_path = images_path + lines[0]
                lines = lines[1:]

                if annotation not in data_map:
                    data_map[annotation] = {}
                    img = Image.open(image_path)
                    data_map[annotation]['filepath'] = image_path
                    data_map[annotation]['width'] = img.width
                    data_map[annotation]['height'] = img.height
                    data_map[annotation]['bboxes'] = []

                for line in lines:
                    line_split = line.strip().split(',')
                    (x1, y1, x2, y2, class_name) = line_split

                    data_map[annotation]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})
        for key in data_map:
            rez.append(data_map[key])
        return rez
