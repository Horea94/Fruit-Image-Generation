from __future__ import division
import numpy as np
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.optimizer_v2.adadelta import Adadelta
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model

import frcnn_config
from utils import simple_parser
from frcnn.frcnn_utils import data_generators, loss_functions
from frcnn.networks import resnet, vgg
from custom_callbacks.CustomModelSaver import CustomModelSaver
from custom_callbacks.CustomModelSaverUtil import CustomModelSaverUtil
# this import is for TF 2.4 compatibility, otherwise the process fails to copy data to the GPU memory
# the import can be replaced with the code that is written in the frcnn_utils/tf_2_4_compatibility.py file


def train(use_saved_rpn=False, model_name='vgg'):
    all_imgs = simple_parser.get_data(frcnn_config.train_annotation_folder, frcnn_config.train_image_folder)

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

    helper = CustomModelSaverUtil()
    best_loss = np.Inf

    data_gen_train = data_generators.CustomDataGenerator(train_imgs, nn.get_img_output_length, batch_size=frcnn_config.batch_size, augment=True, shuffle=True)
    data_gen_val = data_generators.CustomDataGenerator(val_imgs, nn.get_img_output_length, batch_size=frcnn_config.batch_size, augment=False, shuffle=False)

    img_input = Input(shape=frcnn_config.input_shape_img, dtype='float32')

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input)

    # define the RPN, built on the base layers
    rpn = nn.rpn(shared_layers, frcnn_config.num_anchors)

    model_rpn = Model(img_input, rpn[:2])

    rpn_lr = frcnn_config.initial_rpn_lr

    optimizer_rpn = Adadelta(learning_rate=rpn_lr)

    model_rpn.compile(optimizer=optimizer_rpn, loss=[loss_functions.rpn_loss_cls(frcnn_config.num_anchors), loss_functions.rpn_loss_regr(frcnn_config.num_anchors)])

    if use_saved_rpn:
        helper.load_model_weights(model_rpn, frcnn_config.model_path)
        best_loss = helper.load_last_loss(frcnn_config.rpn_loss_path)

    epoch_length = len(data_gen_train)

    print('Starting training')
    model_ckpt = CustomModelSaver(model_path=frcnn_config.model_path, loss_path=frcnn_config.rpn_loss_path, best_loss=best_loss)
    model_lr_monitor = ReduceLROnPlateau(monitor='loss', factor=0.5, min_lr=frcnn_config.min_rpn_lr, patience=10, verbose=1)
    model_rpn.fit(data_gen_train, steps_per_epoch=epoch_length, epochs=frcnn_config.epochs, callbacks=[model_ckpt, model_lr_monitor], verbose=1)


# models currently supported:
# vgg
# resnet
train(use_saved_rpn=False, model_name=frcnn_config.used_model_name)
