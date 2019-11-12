import math
import os

from PIL import Image, ImageEnhance
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adadelta, SGD
from keras_preprocessing.image import ImageDataGenerator

import network
import config
import custom_generator
import util
import random
import numpy as np


def train(load_weights=False):
    gen = custom_generator.CustomDataGenerator(images_path=config.image_folder, masks_path=config.mask_folder, batch_size=config.batch_size, image_dimensions=config.img_size)
    model = network.unet(input_size=config.img_size)
    if load_weights:
        if os.path.exists(config.weight_file):
            model.load_weights(config.weight_file)
            print("Weights successfully loaded from file. Resuming training.")
        else:
            print("Warning! Weight file not present. Beginning training from scratch.")
    model.summary()
    optimizer = Adadelta(lr=1.0)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss=util.dice_coef_multilabel, metrics=[util.tversky_loss, 'accuracy'])
    model_checkpoint = ModelCheckpoint('unet.h5', monitor='loss', mode='min', save_best_only=True, verbose=1)
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, mode='min', verbose=1)
    model.fit_generator(gen, steps_per_epoch=len(gen), epochs=config.epochs, callbacks=[model_checkpoint, learning_rate_reduction], verbose=1)


def test():
    model = network.unet(input_size=config.img_size)
    if os.path.exists(config.weight_file):
        model.load_weights(config.weight_file)
    else:
        print("Warning! Weight file not present.")

    testGen = ImageDataGenerator()

    test_gen = testGen.flow_from_directory(config.test_folder, target_size=config.img_size[:-1], class_mode='sparse',
                                           batch_size=config.batch_size, shuffle=False, subset=None)

    y = model.predict_generator(test_gen, steps=int(math.ceil(test_gen.n / config.batch_size)), verbose=1)
    i = 0
    for item in y:
        img = util.create_output_from_prediction(item)
        img = Image.fromarray(img)
        img.save(config.output_folder + str(i) + '.png')
        i += 1


def test2():
    img = Image.open(config.test_folder + "Input/difficult orange.jpg")
    w, h = img.size
    target_w = round(w / config.img_size[0]) * config.img_size[0]
    target_h = round(h / config.img_size[1]) * config.img_size[1]
    img = np.array(img.resize(size=(target_w, target_h)))
    patches = []
    for i in range(target_h // config.img_size[0]):
        for j in range(target_w // config.img_size[1]):
            patches.append(img[(i * 256):((i + 1) * 256), (j * 256):((j + 1) * 256), :])

    model = network.unet(input_size=config.img_size)
    if os.path.exists(config.weight_file):
        model.load_weights(config.weight_file)
    else:
        print("Warning! Weight file not present.")
    y = model.predict(np.array(patches), verbose=1)
    i = 0
    for item in y:
        img = util.create_output_from_prediction(item)
        img = Image.fromarray(img)
        img.save(config.output_folder + str(i) + '.png')
        i += 1


# train(load_weights=True)
test2()
