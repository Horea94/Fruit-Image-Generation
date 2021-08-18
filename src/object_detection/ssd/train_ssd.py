import math
import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from models.keras_ssd512 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from ssd.custom_callbacks.MyModelCheckpoint import MyModelCheckpoint
from ssd.keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

import ssd_config


def get_previous_epoch_and_loss(filename):
    epoch = 0
    loss = math.inf
    if os.path.exists(filename):
        with open(filename, mode='r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                e, _, v_loss = line.split(',')
                try:
                    e = int(e)
                    v_loss = float(v_loss)
                    if e > epoch:
                        epoch = e
                    if v_loss < loss:
                        loss = v_loss
                except ValueError:
                    pass
        epoch += 1
        print("Loaded previous epoch: %d and loss: %f " % (epoch, loss))
    else:
        print("Training log file not found; using epoch: %d and loss: %f " % (epoch, loss))
    return epoch, loss


def lookup_or_create_h5_dataset(h5_path, imgs_path, annot_path, labels):
    if os.path.exists(h5_path):
        dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=h5_path)
    else:
        dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
        dataset.parse_csv(images_dir=imgs_path,
                          annotations_dir=annot_path,
                          all_labels=labels,
                          input_format=['xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
                          include_classes='all')
        # Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
        # speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
        # option in the constructor, because in that cas the images are in memory already anyway. If you don't
        # want to create HDF5 datasets, comment out the subsequent two function calls.
        dataset.create_hdf5_dataset(file_path=h5_path,
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)
    return dataset


adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

start_epoch = 0
val_loss = math.inf
if ssd_config.use_weights:
    start_epoch, val_loss = get_previous_epoch_and_loss(ssd_config.ssd_training_log)
    model = load_model(ssd_config.ssd_model_path, custom_objects={'AnchorBoxes': AnchorBoxes, 'L2Normalization': L2Normalization, 'compute_loss': ssd_loss.compute_loss})
else:
    if os.path.exists(ssd_config.ssd_training_log):
        os.remove(ssd_config.ssd_training_log)
    model = build_model(image_size=ssd_config.img_shape,
                        n_classes=ssd_config.num_classes - 1,  # num_classes includes the background and build_model requires only the number of positive classes
                        mode='training',
                        l2_regularization=0.0005,
                        scales=ssd_config.scales,
                        aspect_ratios_global=None,
                        aspect_ratios_per_layer=ssd_config.aspect_ratios_per_layer,
                        two_boxes_for_ar1=ssd_config.two_boxes_for_ar1,
                        steps=ssd_config.steps,
                        offsets=ssd_config.offsets,
                        clip_boxes=ssd_config.clip_boxes,
                        variances=ssd_config.variances,
                        normalize_coords=ssd_config.normalize_coords,
                        confidence_thresh=0.5,
                        iou_threshold=0.4)

final_epoch = start_epoch + ssd_config.epochs

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

train_dataset = lookup_or_create_h5_dataset(h5_path=ssd_config.ssd_train_h5_data, imgs_path=ssd_config.train_image_folder, annot_path=ssd_config.train_annotation_folder, labels=ssd_config.fruit_labels)
val_dataset = lookup_or_create_h5_dataset(h5_path=ssd_config.ssd_valid_h5_data, imgs_path=ssd_config.valid_image_folder, annot_path=ssd_config.valid_annotation_folder, labels=ssd_config.fruit_labels)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()
#
print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# 4: Define the image processing chain.

data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 48, 0.5),
                                                            random_contrast=(0.5, 1.8, 0.5),
                                                            random_saturation=(0.5, 1.8, 0.5),
                                                            random_hue=(18, 0.5),
                                                            random_flip=0.5,
                                                            random_translate=((0.03, 0.5), (0.03, 0.5), 0.5),
                                                            random_scale=(0.5, 2.0, 0.5),
                                                            n_trials_max=3,
                                                            clip_boxes=True,
                                                            overlap_criterion='area',
                                                            bounds_box_filter=(0.3, 1.0),
                                                            bounds_validator=(0.5, 1.0),
                                                            n_boxes_min=1,
                                                            background=(0, 0, 0))

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=ssd_config.img_shape[0],
                                    img_width=ssd_config.img_shape[1],
                                    n_classes=ssd_config.num_classes - 1,
                                    predictor_sizes=predictor_sizes,
                                    scales=ssd_config.scales,
                                    aspect_ratios_global=None,
                                    aspect_ratios_per_layer=ssd_config.aspect_ratios_per_layer,
                                    two_boxes_for_ar1=ssd_config.two_boxes_for_ar1,
                                    steps=ssd_config.steps,
                                    offsets=ssd_config.offsets,
                                    clip_boxes=ssd_config.clip_boxes,
                                    variances=ssd_config.variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=ssd_config.normalize_coords)

train_generator = train_dataset.generate(batch_size=ssd_config.batch_size,
                                         shuffle=True,
                                         transformations=[data_augmentation_chain],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=ssd_config.batch_size,
                                     shuffle=False,
                                     transformations=[],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

model_checkpoint = MyModelCheckpoint(filepath=ssd_config.ssd_model_path,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='min',
                                     save_freq='epoch',
                                     best=val_loss)

csv_logger = CSVLogger(filename=ssd_config.ssd_training_log,
                       separator=',',
                       append=True)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=10,
                               verbose=1)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=5,
                                         verbose=1,
                                         min_delta=0.001,
                                         cooldown=0,
                                         min_lr=0.00001)

callbacks = [model_checkpoint,
             csv_logger,
             early_stopping,
             reduce_learning_rate]

history = model.fit(train_generator,
                    steps_per_epoch=math.ceil(train_dataset_size / ssd_config.batch_size),
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=math.ceil(val_dataset_size / ssd_config.batch_size),
                    initial_epoch=start_epoch,
                    epochs=final_epoch)
