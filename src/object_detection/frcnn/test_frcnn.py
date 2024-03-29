from __future__ import division
import os
import cv2
import numpy as np
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model

from frcnn.frcnn_utils import roi_helpers
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


def test(model_name='vgg'):
    img_input = Input(shape=frcnn_config.input_shape_img)
    roi_input = Input(shape=(frcnn_config.num_rois, 4))

    if model_name == 'vgg':
        nn = vgg
    elif model_name == 'resnet':
        nn = resnet
    else:
        print("model with name: %s is not supported" % model_name)
        print("The supported models are:\nvgg\nresnet\n")
        return
    helper = CustomModelSaverUtil()
    feature_map_input = Input(shape=nn.get_feature_maps_shape())

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input)

    # define the RPN, built on the base layers
    num_anchors = len(frcnn_config.anchor_box_scales) * len(frcnn_config.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, frcnn_config.num_rois, nb_classes=frcnn_config.num_classes)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    helper.load_model_weights(model_rpn, frcnn_config.model_path)
    helper.load_model_weights(model_classifier, frcnn_config.model_path)

    bbox_threshold = 0.7

    for img_name in os.listdir(frcnn_config.test_images):
        img_path = frcnn_config.test_images
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
        for jk in range(R.shape[0] // frcnn_config.num_rois + 1):
            ROIs = np.expand_dims(R[frcnn_config.num_rois * jk:frcnn_config.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // frcnn_config.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], frcnn_config.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier.predict([F, ROIs])
            # print(P_cls)

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < 0.6 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = frcnn_config.fruit_labels[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                (x, y, w, h) = ROIs[0, ii, :]

                bboxes[cls_name].append([frcnn_config.rpn_stride * x, frcnn_config.rpn_stride * y, frcnn_config.rpn_stride * (x + w), frcnn_config.rpn_stride * (y + h)])
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

                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (int(frcnn_config.class_to_color[key][0]), int(frcnn_config.class_to_color[key][1]), int(frcnn_config.class_to_color[key][2])), 2)

                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                all_dets.append((key, 100 * new_probs[jk]))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 0.9, 1)
                textOrg = (real_x1, real_y1)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5), (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5), (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 1)
                
                print("X1: %d Y1: %d X2: %d Y2: %d P: %f" % (real_x1, real_y1, real_x2, real_y2, new_probs[jk]))
        # print(all_dets)
        # print(bboxes)
        # enable if you want to show pics

        cv2.imwrite(frcnn_config.output_folder + img_name, img)


test(model_name=frcnn_config.used_model_name)
