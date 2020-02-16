from __future__ import division
import os
import cv2
import numpy as np
import sys
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from utils import roi_helpers
import detection_config
from networks import vgg, resnet

sys.setrecursionlimit(40000)


def format_img_size(img):
    """ formats the image size based on config """
    img_min_side = float(detection_config.im_size)
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


def test(use_vgg=True):
    img_input = Input(shape=detection_config.input_shape_img)
    roi_input = Input(shape=(detection_config.num_rois, 4))

    nn = vgg
    input_shape_features = (None, None, 512)
    model_name_prefix = 'vgg_'
    if not use_vgg:
        nn = resnet
        input_shape_features = (None, None, 1024)
        model_name_prefix = 'resnet_'

    feature_map_input = Input(shape=input_shape_features)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input)

    # define the RPN, built on the base layers
    num_anchors = len(detection_config.anchor_box_scales) * len(detection_config.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, detection_config.num_rois, nb_classes=detection_config.num_classes)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    model_rpn.load_weights(detection_config.models_folder + model_name_prefix + 'test_rpn.h5', by_name=True)
    model_classifier.load_weights(detection_config.models_folder + model_name_prefix + 'test_cls.h5', by_name=True)

    bbox_threshold = 0.8

    for img_name in os.listdir(detection_config.test_folder):
        img_path = detection_config.test_folder
        filepath = os.path.join(img_path, img_name)

        img = cv2.imread(filepath)

        # preprocess image
        X, ratio = format_img(img)
        X = np.transpose(X, (0, 2, 3, 1))
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, overlap_thresh=bbox_threshold)
        print(R.shape)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        for jk in range(R.shape[0] // detection_config.num_rois + 1):
            ROIs = np.expand_dims(R[detection_config.num_rois * jk:detection_config.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // detection_config.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], detection_config.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier.predict([F, ROIs])
            # print(P_cls)

            for ii in range(P_cls.shape[1]):
                (x, y, w, h) = ROIs[0, ii, :]
                print("Predicted %s with probability %f at coords (x0, y0, x1, y1): (%d, %d, %d, %d)" % (detection_config.fruit_labels[np.argmax(P_cls[0, ii, :])], np.max(P_cls[0, ii, :]), 16 * x, 16 * y, 16 * (x + w), 16 * (y + h)))

                if np.max(P_cls[0, ii, :]) < 0.8 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = detection_config.fruit_labels[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                (x, y, w, h) = ROIs[0, ii, :]

                bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            print(key)
            print(len(bboxes[key]))
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=bbox_threshold)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (int(detection_config.class_to_color[key][0]), int(detection_config.class_to_color[key][1]), int(detection_config.class_to_color[key][2])), 2)

                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                all_dets.append((key, 100 * new_probs[jk]))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5), (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5), (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        print(all_dets)
        print(bboxes)
        # enable if you want to show pics

        cv2.imwrite(detection_config.output_folder + img_name, img)


test(use_vgg=True)
