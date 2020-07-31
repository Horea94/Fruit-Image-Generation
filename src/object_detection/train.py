from __future__ import division
import random
import sys
import time
import numpy as np

from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar

import detection_config
from utils import data_generators, loss_functions, simple_parser, roi_helpers
from networks import vgg, resnet
from utils.CustomLearningRateMonitor import CustomLearningRateMonitor
from utils.CustomModelSaverUtil import CustomModelSaverUtil

sys.setrecursionlimit(40000)


def train(use_saved_rpn=False, use_saved_cls=False, model_name='vgg'):
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
        print("model with name: %s is not supported" % model_name)
        print("The supported models are:\nvgg\nresnet\n")
        return
    model_name_prefix = model_name + '_'
    model_path = detection_config.models_folder + model_name_prefix + 'test_model.h5'
    rpn_loss_path = detection_config.models_folder + model_name_prefix + 'loss_rpn'
    cls_loss_path = detection_config.models_folder + model_name_prefix + 'loss_cls'
    helper = CustomModelSaverUtil()

    data_gen_train = data_generators.get_anchor_gt(train_imgs, nn.get_img_output_length, augment=True, shuffle=True)
    data_gen_val = data_generators.get_anchor_gt(val_imgs, nn.get_img_output_length, augment=False, shuffle=False)

    img_input = Input(shape=detection_config.input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input)

    # define the RPN, built on the base layers
    rpn = nn.rpn(shared_layers, detection_config.num_anchors)

    classifier = nn.classifier(shared_layers, roi_input, detection_config.num_rois, nb_classes=detection_config.num_classes)

    model_rpn = Model(img_input, rpn[:2], name='rpn')
    model_cls = Model([img_input, roi_input], classifier, name='cls')
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    rpn_lr = detection_config.initial_rpn_lr
    cls_lr = detection_config.initial_cls_lr

    optimizer_rpn = Adadelta(learning_rate=rpn_lr)
    optimizer_classifier = Adadelta(learning_rate=cls_lr)

    model_rpn.compile(optimizer=optimizer_rpn, loss=[loss_functions.rpn_loss_cls(detection_config.num_anchors), loss_functions.rpn_loss_regr(detection_config.num_anchors)])
    model_cls.compile(optimizer=optimizer_classifier, loss=[loss_functions.class_loss_cls, loss_functions.class_loss_regr(detection_config.num_classes - 1)],
                      metrics={'dense_class_{}'.format(detection_config.num_classes): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    best_rpn_loss = np.Inf
    best_cls_loss = np.Inf

    if use_saved_rpn:
        helper.load_model_weigths(model_rpn, model_path)
        best_rpn_loss = helper.load_last_loss(rpn_loss_path)
    if use_saved_cls:
        helper.load_model_weigths(model_cls, model_path)
        best_cls_loss = helper.load_last_loss(cls_loss_path)

    rpn_monitor = CustomLearningRateMonitor(model=model_rpn, lr=rpn_lr, min_lr=detection_config.min_rpn_lr, reduction_factor=0.5, patience=10)
    cls_monitor = CustomLearningRateMonitor(model=model_cls, lr=cls_lr, min_lr=detection_config.min_cls_lr, reduction_factor=0.5, patience=10)

    epoch_length = len(train_imgs)
    iter_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    print('Starting training')

    for epoch_num in range(detection_config.epochs):
        progbar = Progbar(epoch_length)
        print('Epoch %d/%d' % (epoch_num + 1, detection_config.epochs))

        while True:
            try:
                if len(rpn_accuracy_rpn_monitor) == epoch_length:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print('Average number of overlapping bounding boxes from RPN = %f for %d previous iterations' % (mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
                X, Y, img_data = next(data_gen_train)

                loss_rpn = model_rpn.train_on_batch(x=X, y=Y)

                P_rpn = model_rpn.predict_on_batch(X)
                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], use_regr=True, max_boxes=300, overlap_thresh=0.8)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = np.array([])

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    # continue
                    pos_samples = np.array([])

                if detection_config.num_rois > 1:
                    if len(pos_samples) < detection_config.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, detection_config.num_rois // 2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, detection_config.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, detection_config.num_rois - len(selected_pos_samples), replace=True).tolist()
                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                loss_class = model_cls.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                          ('classifier_cls', np.mean(losses[:iter_num, 2])), ('classifier_regr', np.mean(losses[:iter_num, 3])),
                                          ("average number of objects", len(selected_pos_samples))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    rpn_monitor.reduce_lr_on_plateau(loss_rpn_cls + loss_rpn_regr)
                    cls_monitor.reduce_lr_on_plateau(loss_class_cls + loss_class_regr)

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: %f' % mean_overlapping_bboxes)
                    print('Classifier accuracy for bounding boxes from RPN: %f ' % class_acc)
                    print('Loss RPN cls: %f' % loss_rpn_cls)
                    print('Loss RPN regr: %f' % loss_rpn_regr)
                    print('Loss Classifier cls: %f' % loss_class_cls)
                    print('Loss Classifier regr: %f' % loss_class_regr)
                    print('Elapsed time: %f' % (time.time() - start_time))

                    curr_rpn_loss = loss_rpn_cls + loss_rpn_regr
                    curr_cls_loss = loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_rpn_loss + curr_cls_loss < best_rpn_loss + best_cls_loss:
                        print('Total loss for model decreased from %f to %f, saving weights' % (best_rpn_loss + best_cls_loss, curr_rpn_loss + curr_cls_loss))
                        best_rpn_loss = curr_rpn_loss
                        best_cls_loss = curr_cls_loss
                        helper.save_model_weights(model_all, model_path)
                        helper.save_loss(curr_rpn_loss, rpn_loss_path)
                        helper.save_loss(curr_cls_loss, cls_loss_path)
                    else:
                        print('Total loss for model did not improve from %f' % (best_rpn_loss + best_cls_loss))

                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                continue

    print('Training complete, exiting.')


train(use_saved_rpn=True, use_saved_cls=True, model_name=detection_config.used_model_name)
