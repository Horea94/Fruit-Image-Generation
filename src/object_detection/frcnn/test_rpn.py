from __future__ import division
import cv2
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from utils import simple_parser
from frcnn.frcnn_utils import roi_helpers, data_generators
import frcnn_config
from frcnn.networks import resnet, vgg
from custom_callbacks.CustomModelSaverUtil import CustomModelSaverUtil
# this import is for TF 2.4 compatibility, otherwise the process fails to copy data to the GPU memory
# the import can be replaced with the code that is written in the frcnn_utils/tf_2_4_compatibility.py file


def format_img_size(img):
    """ formats the image size based on config """
    img_min_side = float(frcnn_config.img_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img)
    img = format_img_channels(img)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


# Currently this outputs a list of  bounding boxes that the rpn predicts
# TODO: filter the predicted bounding boxes to see if any overlap with the ground truth
def test(model_name='vgg'):
    img_input = Input(shape=frcnn_config.input_shape_img)

    if model_name == 'vgg':
        nn = vgg
    elif model_name == 'resnet':
        nn = resnet
    else:
        print("model with name: %s is not supported" % model_name)
        print("The supported models are:\nvgg\nresnet\n")
        return

    helper = CustomModelSaverUtil()

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input)

    # define the RPN, built on the base layers
    num_anchors = len(frcnn_config.anchor_box_scales) * len(frcnn_config.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    model_rpn = Model(img_input, rpn_layers)
    helper.load_model_weights(model_rpn, frcnn_config.model_path)

    bbox_threshold = 0.8
    all_img_data = simple_parser.get_data(frcnn_config.test_annotations, frcnn_config.test_images)

    for img_data in all_img_data:
        height, width, resized_height, resized_width, img_data_aug, x_img = data_generators.augment_and_resize_image(img_data, augment=False)
        y_rpn_cls, y_rpn_regr = data_generators.calc_rpn(img_data_aug, width, height, resized_width, resized_height, nn.get_img_output_length)
        x_img, y_rpn_cls, y_rpn_regr = data_generators.arrange_dims(x_img, y_rpn_cls, y_rpn_regr)

        # img = cv2.imread(img_data['filepath'])

        # preprocess image
        # X, ratio = format_img(img)
        # X = np.transpose(X, (0, 2, 3, 1))
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(x_img)
        print("Image name: %s" % img_data['filepath'])
        print("Expected: ")
        for l in img_data['bboxes']:
            print(l)
        print("Predicted: ")
        R = roi_helpers.rpn_to_roi(Y1, Y2, overlap_thresh=bbox_threshold)
        for i in range(R.shape[0]):
            (x, y, x2, y2) = R[i, :]
            print((x*frcnn_config.rpn_stride, y*frcnn_config.rpn_stride, x2*frcnn_config.rpn_stride, y2*frcnn_config.rpn_stride))


test(model_name=frcnn_config.used_model_name)
