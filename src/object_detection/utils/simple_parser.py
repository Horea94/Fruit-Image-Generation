import cv2
import numpy as np
import os
import detection_config


def get_data(annotations_path, images_path):
    all_imgs = {}

    print('Parsing annotation files')
    for annotation_file in os.listdir(annotations_path):

        with open(annotations_path + annotation_file, 'r') as f:

            lines = [k.strip() for k in f.readlines()]
            filename = images_path + lines[0]
            lines = lines[1:]
            for line in lines:
                line_split = line.strip().split(',')
                (x1, y1, x2, y2, class_name) = line_split
                class_index = detection_config.fruit_labels.index(class_name)

                if filename not in all_imgs:
                    all_imgs[filename] = {}

                    img = cv2.imread(filename)
                    (rows, cols) = img.shape[:2]
                    all_imgs[filename]['filepath'] = filename
                    all_imgs[filename]['width'] = cols
                    all_imgs[filename]['height'] = rows
                    all_imgs[filename]['bboxes'] = []
                    # if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                    # else:
                    #     all_imgs[filename]['imageset'] = 'test'

                all_imgs[filename]['bboxes'].append({'class': class_index, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

            all_data = []
            for key in all_imgs:
                all_data.append(all_imgs[key])

    return all_data
