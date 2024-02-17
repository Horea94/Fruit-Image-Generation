import os
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

# Change these imports to use either ssd300 or ssd512
# ---------------- SSD 300 ----------------
# from models.keras_ssd300 import build_model
# import ssd300_config as ssd_config
# ---------------- SSD 512 ----------------
from models.keras_ssd512 import build_model
import ssd512_config as ssd_config

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
np.set_printoptions(precision=2, suppress=True, linewidth=90)

# batch_images = []
# batch_images.append(np.asarray(img, dtype=np.uint8))
# batch_images = np.array(batch_images)

colors = plt.cm.get_cmap('viridis', ssd_config.num_classes)  # Set the colors for the bounding boxes

fruit_count_per_img = {}
if os.path.exists(ssd_config.test_annotations):
    for file in os.listdir(ssd_config.test_annotations):
        with open(ssd_config.test_annotations + file, mode='r') as f:
            lines = f.readlines()
            filename = lines[0].strip()
            lines = lines[1:]
            fruit_count_per_img[filename] = len(lines)

global_acc = 0

if not os.path.exists(ssd_config.output_folder):
    os.makedirs(ssd_config.output_folder)

for file in os.listdir(ssd_config.test_images):
    imgs = []
    img = Image.open(ssd_config.test_images + file)
    original_w, original_h = img.size
    h, w = ssd_config.img_shape[:2]
    img = img.resize((w, h))
    img = np.asarray(img, dtype=np.uint8)
    imgs.append(img)
    y_pred = model.predict(np.array(imgs))
    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] >= 0.5] for k in range(y_pred.shape[0])]

    print(file)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh)

    for prediction in y_pred_thresh:
        for box in prediction:
            xmin = int(box[-4])
            ymin = int(box[-3])
            xmax = int(box[-2])
            ymax = int(box[-1])
            color = colors(int(box[0]) * 1 / ssd_config.num_classes)
            label = '{:.2f}'.format(box[1])
            # current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            # current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})

            cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(255, 0, 0), thickness=5)

            # (retval, baseLine) = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
            # textOrg = (xmin, ymin)
            # cv2.rectangle(img, (np.float32(textOrg[0] - 5), np.float32(textOrg[1] + baseLine - 5)), (np.float32(textOrg[0] + retval[0] + 5), np.float32(textOrg[1] - retval[1] - 5)), (0, 0, 0), 2)
            # cv2.rectangle(img, (np.float32(textOrg[0] - 5), np.float32(textOrg[1] + baseLine - 5)), (np.float32(textOrg[0] + retval[0] + 5), np.float32(textOrg[1] - retval[1] - 5)), (255, 255, 255), -1)
            cv2.putText(img, label, (xmin - 10, ymin - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 2)

    if file in fruit_count_per_img:
        global_acc += (1 - abs(len(y_pred_thresh[0]) - fruit_count_per_img[file])/fruit_count_per_img[file])

    img = Image.fromarray(img)
    img = img.resize((original_w, original_h))
    img.save(ssd_config.output_folder + file)

print('Acc: %f' % (global_acc/len(fruit_count_per_img)))