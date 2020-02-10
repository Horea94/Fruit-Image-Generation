from __future__ import division
import sys
import tensorflow as tf
import numpy as np
from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import Adadelta
from keras.layers import Input
from keras.models import Model
from keras.backend.tensorflow_backend import set_session

import detection_config
from utils import data_generators, loss_functions, simple_parser
from networks import vgg, resnet
from custom_callbacks.CustomModelSaver import CustomModelSaver
from utils.CustomModelSaverUtil import CustomModelSaverUtil

config2 = tf.ConfigProto()
config2.gpu_options.allow_growth = True
set_session(tf.Session(config=config2))

sys.setrecursionlimit(40000)


def train(use_saved_rpn=False, use_vgg=True):
    all_imgs = simple_parser.get_data(detection_config.annotation_folder, detection_config.image_folder)

    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))

    nn = vgg
    model_name_prefix = 'vgg_'
    if not use_vgg:
        nn = resnet
        model_name_prefix = 'resnet_'
    model_path = detection_config.models_folder + model_name_prefix + 'test_rpn.h5'
    loss_path = detection_config.models_folder + model_name_prefix + 'loss_rpn'
    helper = CustomModelSaverUtil()
    best_loss = np.Inf

    data_gen_train = data_generators.get_anchor_gt(train_imgs, nn.get_img_output_length, mode='train')
    data_gen_val = data_generators.get_anchor_gt(val_imgs, nn.get_img_output_length, mode='val')

    img_input = Input(shape=detection_config.input_shape_img)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input)

    # define the RPN, built on the base layers
    rpn = nn.rpn(shared_layers, detection_config.num_anchors)

    model_rpn = Model(img_input, rpn[:2])

    rpn_lr = detection_config.initial_rpn_lr

    optimizer_rpn = Adadelta(lr=rpn_lr)

    model_rpn.compile(optimizer=optimizer_rpn, loss=[loss_functions.rpn_loss_cls(detection_config.num_anchors), loss_functions.rpn_loss_regr(detection_config.num_anchors)])

    if use_saved_rpn:
        helper.load_model_weigths(model_rpn, model_path)
        best_loss = helper.load_last_loss(loss_path)

    epoch_length = len(train_imgs)

    print('Starting training')
    model_ckpt = CustomModelSaver(model_path=model_path, loss_path=loss_path, best_loss=best_loss)
    model_lr_monitor = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
    model_rpn.fit_generator(data_gen_train, steps_per_epoch=epoch_length, epochs=detection_config.epochs, callbacks=[model_ckpt, model_lr_monitor], verbose=1)


train(use_saved_rpn=True, use_vgg=True)
