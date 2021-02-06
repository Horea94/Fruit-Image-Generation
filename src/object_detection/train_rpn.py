from __future__ import division
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import detection_config
from utils import data_generators, loss_functions, simple_parser
from networks import vgg, resnet
from custom_callbacks.CustomModelSaver import CustomModelSaver
from utils.CustomModelSaverUtil import CustomModelSaverUtil
# this import is for TF 2.4 compatibility, otherwise the process fails to copy data to the GPU memory
# the import can be replaced with the code that is written in the utils/tf_2_4_compatibility.py file
import utils.tf_2_4_compatibility


def train(use_saved_rpn=False, model_name='vgg'):
    all_imgs = simple_parser.get_data(detection_config.annotation_folder, detection_config.image_folder)

    train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'val']

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))

    if model_name == 'vgg':
        nn = vgg
    elif model_name == 'resnet':
        nn = resnet
    else:
        print("Model with name: %s is not supported" % model_name)
        print("The supported models are:\nvgg\nresnet\n")
        return
    model_name_prefix = model_name + '_'

    model_path = detection_config.models_folder + model_name_prefix + 'test_model.h5'
    loss_path = detection_config.models_folder + model_name_prefix + 'loss_rpn'
    helper = CustomModelSaverUtil()
    best_loss = np.Inf

    data_gen_train = data_generators.CustomDataGenerator(train_imgs, nn.get_img_output_length, batch_size=detection_config.batch_size, augment=True, shuffle=True)
    data_gen_val = data_generators.CustomDataGenerator(val_imgs, nn.get_img_output_length, batch_size=detection_config.batch_size, augment=False, shuffle=False)

    img_input = Input(shape=detection_config.input_shape_img, dtype='float32')

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input)

    # define the RPN, built on the base layers
    rpn = nn.rpn(shared_layers, detection_config.num_anchors)

    model_rpn = Model(img_input, rpn[:2])

    rpn_lr = detection_config.initial_rpn_lr

    optimizer_rpn = Adadelta(learning_rate=rpn_lr)

    model_rpn.compile(optimizer=optimizer_rpn, loss=[loss_functions.rpn_loss_cls(detection_config.num_anchors), loss_functions.rpn_loss_regr(detection_config.num_anchors)])

    if use_saved_rpn:
        helper.load_model_weigths(model_rpn, model_path)
        best_loss = helper.load_last_loss(loss_path)

    epoch_length = len(data_gen_train)

    print('Starting training')
    model_ckpt = CustomModelSaver(model_path=model_path, loss_path=loss_path, best_loss=best_loss)
    model_lr_monitor = ReduceLROnPlateau(monitor='loss', factor=0.5, min_lr=detection_config.min_rpn_lr, patience=10, verbose=1)
    model_rpn.fit(data_gen_train, steps_per_epoch=epoch_length, epochs=detection_config.epochs, callbacks=[model_ckpt, model_lr_monitor], verbose=1)


# models currently supported:
# vgg
# resnet
train(use_saved_rpn=True, model_name=detection_config.used_model_name)
