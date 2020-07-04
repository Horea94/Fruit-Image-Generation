import numpy as np


class DatasetStats:
    def __init__(self):
        self.minimum_width_img_w = np.Inf
        self.minimum_width_img_h = np.Inf
        self.minimum_height_img_w = np.Inf
        self.minimum_height_img_h = np.Inf
        self.minimum_area = np.Inf
        self.minimum_area_img_w = np.Inf
        self.minimum_area_img_h = np.Inf