import json
import os
import math
import ntpath
import detection_config


def round_number(x):
    if x - int(x) >= 0.5:
        return math.ceil(x)
    else:
        return math.floor(x)


for filename in os.listdir(detection_config.dataset_root + "Test/manual_annotations/"):
    json_filename = detection_config.dataset_root + "Test/manual_annotations/" + filename
    with open(json_filename) as json_file:
        json_data = json.load(json_file)
        img_path = json_data['imagePath'].strip('/\\')
        img_name_ext = ntpath.basename(img_path)
        img_name = img_name_ext.split('.')[0]
        with open(detection_config.test_annotations + img_name + ".csv", 'w') as f:
            f.write(img_name_ext + '\n')
            for shape in json_data['shapes']:
                fruit_label = shape['label']
                points = shape['points']
                f.write(str(round_number(points[0][0])) + ',' + str(round_number(points[0][1])) + ',' +
                        str(round_number(points[1][0])) + ',' + str(round_number(points[1][1])) + ',' + fruit_label + '\n')
