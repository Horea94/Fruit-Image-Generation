import math
import os

from PIL import Image, ImageEnhance
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import Adadelta, SGD
from keras_preprocessing.image import ImageDataGenerator

import network
import config
import custom_generator
import util
import numpy as np


def train(load_weights=False):
    gen = custom_generator.CustomDataGenerator(images_path=config.image_folder, masks_path=config.mask_folder, batch_size=config.batch_size, image_dimensions=config.img_size)
    model = network.unet(input_size=config.img_size)
    if load_weights:
        if os.path.exists(config.unet_weights):
            model.load_weights(config.unet_weights)
            print("Weights successfully loaded from file. Resuming training.")
        else:
            print("Warning! Weight file not present. Beginning training from scratch.")
    model.summary()
    optimizer = Adadelta(lr=0.1)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # experimental_run_tf_function set to false to avoid an error caused by the way tversky_loss is defined
    model.compile(optimizer=optimizer, loss=util.dice_coef_multilabel, metrics=[util.tversky_loss, 'accuracy'], experimental_run_tf_function=False)
    model_checkpoint = ModelCheckpoint(config.unet_weights, monitor='loss', mode='min', save_best_only=True, verbose=1)
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, mode='min', verbose=1)
    model.fit(gen, steps_per_epoch=len(gen), epochs=config.epochs, callbacks=[model_checkpoint, learning_rate_reduction], verbose=1)


def test():
    model = network.unet(input_size=config.img_size)
    if os.path.exists(config.unet_weights):
        model.load_weights(config.unet_weights)
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
    img = Image.open(config.test_folder + "6.png")
    w, h = img.size
    target_w = round(w / config.img_size[0]) * config.img_size[0]
    target_h = round(h / config.img_size[1]) * config.img_size[1]
    img = np.array(img.resize(size=(target_w, target_h)))
    patches = []
    for i in range(target_h // config.img_size[0]):
        for j in range(target_w // config.img_size[1]):
            patches.append(img[(i * 256):((i + 1) * 256), (j * 256):((j + 1) * 256), :])

    model = network.unet(input_size=config.img_size)
    if os.path.exists(config.unet_weights):
        model.load_weights(config.unet_weights)
    else:
        print("Warning! Weights file not present.")
    preds = model.predict(np.array(patches), verbose=1)
    for i in range(target_h // config.img_size[0]):
        for j in range(target_w // config.img_size[1]):
            pred = preds[i*(target_w // config.img_size[1]) + j]
            item = util.create_output_from_prediction(pred)
            for x in range(config.img_size[0]):
                for y in range(config.img_size[1]):
                    if (item[x][y] != 0).all():
                        img[i*config.img_size[0] + x][j*config.img_size[1] + y] = item[x][y]
    img = Image.fromarray(img).resize(size=(w, h))
    img.save(config.output_folder + 'rez.png')


train(load_weights=False)
# test2()
