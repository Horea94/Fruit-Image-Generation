import numpy as np
from keras import backend as K
import config


def create_target_from_mask(mask_img):
    target_shape = mask_img.shape[:-1] + (config.num_classes,)
    encoded_image = np.zeros(target_shape, dtype=np.int8)
    for i in range(config.num_classes):
        encoded_image[:, :, i] = np.all(mask_img.reshape((-1, 3)) == config.color_map[i], axis=1).reshape(target_shape[:-1])
    return encoded_image


def create_output_from_prediction(prediction):
    single_layer = np.argmax(prediction, axis=-1)
    output = np.zeros(prediction.shape[:2] + (3,))
    for k in range(config.num_classes):
        output[single_layer == k] = config.color_map[k]
    return np.uint8(output)


def dice_coef(y_true, y_pred, index, smooth=1e-10):
    # import tensorflow as tf
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # print_op = tf.Print(dice, [dice, tf.shape(y_true_f), tf.shape(y_pred_f)], message=config.labels[index])
    return dice


def dice_coef_multilabel(y_true, y_pred):
    dice = 0
    # calculate dice for each class excluding the background class
    for index in range(config.num_classes):
        dice -= dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index], index)
    return dice


def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # probe that voxels are class i
    p1 = ones - y_pred  # probe that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0)
    den = num + alpha * K.sum(p0 * g1) + beta * K.sum(p1 * g0)

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T
