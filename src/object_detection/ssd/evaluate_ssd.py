import ssd_config
from models.keras_ssd512 import build_model
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator
from matplotlib import pyplot as plt
import numpy as np

model = build_model(image_size=ssd_config.img_shape,
                    n_classes=ssd_config.num_classes - 1,
                    mode='inference',
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

model.load_weights(ssd_config.ssd_model_path, by_name=True)

dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
dataset.parse_csv(images_dir=ssd_config.test_images,
                  annotations_dir=ssd_config.test_annotations,
                  all_labels=ssd_config.fruit_labels,
                  input_format=['xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
                  include_classes='all')

evaluator = Evaluator(model=model,
                      n_classes=ssd_config.num_classes - 1,
                      data_generator=dataset,
                      model_mode='inference')

results = evaluator(img_height=ssd_config.img_shape[0],
                    img_width=ssd_config.img_shape[1],
                    batch_size=ssd_config.batch_size,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='integrate',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results

for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(ssd_config.fruit_labels[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))

plt.figure(figsize=(10, 10))
plt.plot(recalls[1], precisions[1], color='blue', linewidth=1.5)
plt.xlabel('recall', fontsize=16)
plt.ylabel('precision', fontsize=16)
plt.grid(True)
plt.xticks(np.linspace(0, 1, 11), fontsize=16)
plt.yticks(np.linspace(0, 1, 11), fontsize=16)
plt.title("{}, AP@0.5 IoU: {:.3f}".format(ssd_config.fruit_labels[1], average_precisions[1]), fontsize=16)
plt.show()

recall = recalls[1][-1]
precision = precisions[1][-1]
f1 = 2.0 * recall * precision / (precision + recall)
print("recall: %f - precision: %f - F1: %f" % (recall, precision, f1))