import math
import os
import glob
import cv2
import threading
import numpy as np
import random
import xml.etree.ElementTree as ep
import detection_config as config
from shapely.geometry import Polygon
from utils.DatasetStats import DatasetStats

stats = DatasetStats()


def build_dataset(image_save_path, mask_save_path, annotation_save_path, thread_id, total_threads, limit, mutex, is_binary_mask=True, simple_annotation_format=True):
    global stats
    local_stats = DatasetStats()
    bkg_image_paths = [config.background_folder + x for x in os.listdir(config.background_folder)]
    labels_to_images = {}
    rotation_angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    mutex.acquire()
    try:
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)
        if not os.path.exists(annotation_save_path):
            os.makedirs(annotation_save_path)
    finally:
        mutex.release()

    for i in range(1, len(config.fruit_labels)):
        label = config.fruit_labels[i]
        labels_to_images[label] = glob.glob(config.source_dataset_train_folder + label + '/*')

    for index in range(limit):
        skewed_width = math.floor(config.img_shape[1] * random.uniform(config.img_skew_range[0], config.img_skew_range[1]))
        skewed_height = math.floor(config.img_shape[0] * random.uniform(config.img_skew_range[0], config.img_skew_range[1]))
        img_count = index * total_threads + thread_id
        canvas = cv2.imread(bkg_image_paths[random.randint(0, len(bkg_image_paths) - 1)])
        canvas = cv2.resize(canvas, (skewed_width, skewed_height))
        canvas = enhance_image(canvas)
        mask_canvas = np.zeros((skewed_height, skewed_width, config.img_shape[2]), dtype=np.uint8)
        anchors = []
        # fruits_in_image = random.randint(1, 6)
        fruits_in_image = 15
        for i in range(fruits_in_image):
            fruit_label_index = random.randint(1, len(config.fruit_labels) - 1)
            fruit_label = config.fruit_labels[fruit_label_index]
            fruit_image_path = labels_to_images[fruit_label][random.randint(0, len(labels_to_images[fruit_label]) - 1)]
            fruit_img_size = random.randint(config.min_fruit_size, config.max_fruit_size)
            rotate_index = random.randint(0, 3)
            fruit_image = cv2.imread(fruit_image_path)
            fruit_image = cv2.resize(fruit_image, (fruit_img_size, fruit_img_size))
            if rotate_index < 3:
                fruit_image = cv2.rotate(fruit_image, rotateCode=rotation_angles[rotate_index])
            fruit_mask = build_mask(fruit_image)
            non_empty_cols = np.where(np.amax(fruit_mask, axis=0) > 0)[0]
            non_empty_rows = np.where(np.amax(fruit_mask, axis=1) > 0)[0]
            top_most_px = min(non_empty_rows)
            bottom_most_px = max(non_empty_rows)
            left_most_px = min(non_empty_cols)
            right_most_px = max(non_empty_cols)
            fruit_mask = fruit_mask[top_most_px:bottom_most_px + 1, left_most_px:right_most_px + 1]
            fruit_image = fruit_image[top_most_px:bottom_most_px + 1, left_most_px:right_most_px + 1]
            w, h = fruit_image.shape[:2]
            if min(w, h) < config.min_fruit_size or max(w, h) > config.max_fruit_size:
                ratio = min(config.min_fruit_size / min(w, h), config.max_fruit_size / max(w, h))
                fruit_image = cv2.resize(fruit_image, (int(h * ratio), int(w * ratio)))
                fruit_mask = build_mask(fruit_image)
            fruit_image = enhance_image(fruit_image)
            # pad the image by a percentage so the resulting bounding box is slightly bigger than the fruit
            w, h = fruit_image.shape[:2]
            fruit_image = cv2.copyMakeBorder(fruit_image, top=int(h * config.bounding_box_padding), bottom=int(h * config.bounding_box_padding), left=int(w * config.bounding_box_padding),
                                             right=int(w * config.bounding_box_padding), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            fruit_mask = cv2.copyMakeBorder(fruit_mask, top=int(h * config.bounding_box_padding), bottom=int(h * config.bounding_box_padding), left=int(w * config.bounding_box_padding),
                                             right=int(w * config.bounding_box_padding), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            if not is_binary_mask:
                fruit_mask = color_mask(fruit_mask, config.color_map[fruit_label_index])
            successfully_added_img, w, h = add_image_and_mask_to_canvas(canvas, fruit_image, mask_canvas, fruit_mask, anchors, fruit_label)
            if successfully_added_img:
                update_stats(local_stats, h, w)
        canvas = cv2.resize(canvas, (config.img_shape[1], config.img_shape[0]))
        mask_canvas = cv2.resize(mask_canvas, (config.img_shape[1], config.img_shape[0]))
        cv2.imwrite(image_save_path + str(img_count) + '.png', canvas)
        cv2.imwrite(mask_save_path + str(img_count) + '.png', mask_canvas)
        anchors = resize_bounding_boxes(anchors, skewed_width, skewed_height, config.img_shape[1], config.img_shape[0])
        write_annotation_to_file(annotation_save_path, anchors, img_count, simple_format=simple_annotation_format)
        print("Thread %d saved image %d.png" % (thread_id, img_count))
    mutex.acquire()
    if local_stats.minimum_area < stats.minimum_area:
        stats.minimum_area_img_w = local_stats.minimum_area_img_w
        stats.minimum_area_img_h = local_stats.minimum_area_img_h
        stats.minimum_area = local_stats.minimum_area
    if local_stats.minimum_height_img_h < stats.minimum_height_img_h:
        stats.minimum_height_img_h = local_stats.minimum_height_img_h
        stats.minimum_height_img_w = local_stats.minimum_height_img_w
    if local_stats.minimum_width_img_w < stats.minimum_width_img_w:
        stats.minimum_width_img_h = local_stats.minimum_width_img_h
        stats.minimum_width_img_w = local_stats.minimum_width_img_w
    mutex.release()


def update_stats(stats_param, h, w):
    if stats_param.minimum_area > w * h:
        stats_param.minimum_area_img_h = h
        stats_param.minimum_area_img_w = w
        stats_param.minimum_area = w * h
    if stats_param.minimum_height_img_h > h:
        stats_param.minimum_height_img_h = h
        stats_param.minimum_height_img_w = w
    if stats_param.minimum_width_img_w > w:
        stats_param.minimum_width_img_h = h
        stats_param.minimum_width_img_w = w


def resize_bounding_boxes(anchors, old_width, old_height, new_width, new_height):
    resized_anchors = []
    for anchor in anchors:
        resized_anchors.append(((round_number(new_height * anchor[0][0] / old_height), round_number(new_width * anchor[0][1] / old_width)),
                                (round_number(new_height * anchor[1][0] / old_height), round_number(new_width * anchor[1][1] / old_width)),
                                (round_number(new_height * anchor[2][0] / old_height), round_number(new_width * anchor[2][1] / old_width)),
                                (round_number(new_height * anchor[3][0] / old_height), round_number(new_width * anchor[3][1] / old_width)),
                                anchor[4]))
    return resized_anchors


def round_number(x):
    if x - int(x) >= 0.5:
        return math.ceil(x)
    else:
        return math.floor(x)


def write_annotation_to_file(annotation_save_path, anchors, img_count, simple_format=True):
    if simple_format:
        with open(annotation_save_path + str(img_count), 'w') as f:
            f.write(str(img_count) + '.png\n')
            for anchor in anchors:
                f.write(str(anchor[0][1]) + ',' + str(anchor[0][0]) + ',' + str(anchor[3][1]) + ',' + str(anchor[3][0]) + ',' + anchor[4] + '\n')
    else:
        root = ep.Element('annotation')
        ep.SubElement(root, 'path').text = config.train_image_folder + str(img_count) + '.png'
        for anchor in anchors:
            obj = ep.SubElement(root, 'object')
            ep.SubElement(obj, 'name').text = anchor[4]
            bndbox = ep.SubElement(obj, 'bndbox')
            ep.SubElement(bndbox, 'xmin').text = str(anchor[0][1])
            ep.SubElement(bndbox, 'xmax').text = str(anchor[3][1])
            ep.SubElement(bndbox, 'ymin').text = str(anchor[0][0])
            ep.SubElement(bndbox, 'ymax').text = str(anchor[3][0])
        tree = ep.ElementTree(root)
        tree.write(file_or_filename=annotation_save_path + str(img_count) + '.xml')


def enhance_image(canvas, contrast=True, brightness=True):
    brightness_factor = 0
    contrast_factor = 1.0
    if contrast:
        contrast_factor = random.random() * 0.4 + 0.8
    if brightness:
        brightness_factor = random.random() * 0.4 + 0.8
    canvas = cv2.convertScaleAbs(canvas, alpha=contrast_factor, beta=brightness_factor)
    return canvas


# TODO: add partial occlusion
def add_image_and_mask_to_canvas(canvas, fruit_image, canvas_mask, fruit_mask, anchors, fruit_label):
    # bounds inside of which the fruit image can be added to the canvas
    # the fruit image could be partially outside of the canvas, to emulate the case where only part of the fruit is visible in an image
    # max_x = canvas.shape[0] - fruit_image.shape[0] // 2
    # max_y = canvas.shape[1] - fruit_image.shape[1] // 2
    # min_x = -fruit_image.shape[0]
    # min_y = -fruit_image.shape[1]
    max_x = canvas.shape[0] - fruit_image.shape[0] - 1
    max_y = canvas.shape[1] - fruit_image.shape[1] - 1
    min_x = 0
    min_y = 0
    x = 0
    y = 0
    done = False
    attempts = 10
    if min_x > max_x or min_y > max_y:
        return done, 0, 0
    while not done and attempts > 0:
        # attempt to find a free area on the canvas to add the image
        # if no free space is found, the image is not added
        x = random.randint(min_x // 2, max_x)
        y = random.randint(min_y // 2, max_y)
        done = not is_overlap_between_new_image_and_old_images(((x, y), (x, y + fruit_image.shape[1]), (x + fruit_image.shape[0], y), (x + fruit_image.shape[0], y + fruit_image.shape[1])), anchors)
        attempts -= 1
    if done:
        for i in range(fruit_image.shape[0]):
            for j in range(fruit_image.shape[1]):
                if 0 <= x + i < canvas.shape[0] and 0 <= y + j < canvas.shape[1]:
                    if (fruit_mask[i][j] == 255).all():
                        canvas[x + i][y + j] = fruit_image[i][j]
                        canvas_mask[x + i][y + j] = fruit_mask[i][j]
        # if the fruit is only partially included in the image, set the anchor bounds to the edge of the canvas
        upper_left = (max(x, 0), max(y, 0))
        lower_left = (max(x, 0), min(y + fruit_image.shape[1], canvas.shape[1] - 1))
        upper_right = (min(x + fruit_image.shape[0], canvas.shape[0] - 1), max(y, 0))
        lower_right = (min(x + fruit_image.shape[0], canvas.shape[0] - 1), min(y + fruit_image.shape[1], canvas.shape[1] - 1))
        anchors.append((upper_left, lower_left, upper_right, lower_right, fruit_label))
    return done, fruit_image.shape[0], fruit_image.shape[1]


def is_overlap_between_new_image_and_old_images(img_coordinates, other_images):
    for old_img_coords in other_images:
        if is_overlap_between_images(img_coordinates, old_img_coords):
            return True
    return False


def is_overlap_between_images(src_img_coordinates, dest_img_coordinates):
    upper_left_point = src_img_coordinates[0]
    upper_right_point = src_img_coordinates[1]
    lower_left_point = src_img_coordinates[2]
    lower_right_point = src_img_coordinates[3]

    upper_left_point_dest = dest_img_coordinates[0]
    upper_right_point_dest = dest_img_coordinates[1]
    lower_left_point_dest = dest_img_coordinates[2]
    lower_right_point_dest = dest_img_coordinates[3]

    # adjust the coordinates of each image using a factor to allow some degree of overlap between them
    # source image
    factor = config.overlap_factor
    lower_left_point, lower_right_point, upper_left_point, upper_right_point = adjust_bounds_for_overlap(factor, upper_left_point, upper_right_point, lower_left_point, lower_right_point)
    # destination image
    lower_left_point_dest, lower_right_point_dest, upper_left_point_dest, upper_right_point_dest = adjust_bounds_for_overlap(factor, upper_left_point_dest, upper_right_point_dest, lower_left_point_dest,
                                                                                                                             lower_right_point_dest)

    p1 = Polygon([upper_left_point, upper_right_point, lower_right_point, lower_left_point, upper_left_point])
    p2 = Polygon([upper_left_point_dest, upper_right_point_dest, lower_right_point_dest, lower_left_point_dest, upper_left_point_dest])
    return p1.intersects(p2)


def adjust_bounds_for_overlap(factor, upper_left_point, upper_right_point, lower_left_point, lower_right_point):
    src_img_width = abs(upper_left_point[0] - lower_left_point[0])
    src_img_height = abs(upper_left_point[1] - upper_right_point[1])
    # as the upper left corner of an image is (0, 0), the upper bound will be smaller than the lower bound
    upper_left_point = (upper_left_point[0] + math.floor(src_img_width * factor), upper_left_point[1] + math.floor(src_img_height * factor))
    upper_right_point = (upper_right_point[0] + math.floor(src_img_width * factor), upper_right_point[1] - math.floor(src_img_height * factor))
    lower_left_point = (lower_left_point[0] - math.floor(src_img_width * factor), lower_left_point[1] + math.floor(src_img_height * factor))
    lower_right_point = (lower_right_point[0] - math.floor(src_img_width * factor), lower_right_point[1] - math.floor(src_img_height * factor))
    return lower_left_point, lower_right_point, upper_left_point, upper_right_point


def build_mask(fruit_image, threshold=config.mask_threshold):
    img = cv2.cvtColor(fruit_image, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    img_flood_fill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img_flood_fill, mask, (0, 0), 255)
    cv2.floodFill(img_flood_fill, mask, (w - 1, 0), 255)
    cv2.floodFill(img_flood_fill, mask, (0, h - 1), 255)
    cv2.floodFill(img_flood_fill, mask, (w - 1, h - 1), 255)
    img_flood_fill = cv2.bitwise_not(img_flood_fill)
    img = img_flood_fill | img
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1)
    return img


def color_mask(fruit_mask, fruit_mask_color):
    for i in range(fruit_mask.shape[0]):
        for j in range(fruit_mask.shape[1]):
            if (fruit_mask[i][j] == 255).all():
                fruit_mask[i][j] = fruit_mask_color
    return fruit_mask


if __name__ == "__main__":
    thrd_list = []
    train_mutex = threading.Lock()
    valid_mutex = threading.Lock()
    image_limit = config.train_dataset_generation_limit
    if image_limit > 0:
        max_images_per_thread = int(math.ceil(image_limit / config.total_threads))
        for index in range(config.total_threads):
            thread = threading.Thread(target=build_dataset, args=(config.train_image_folder, config.train_mask_folder, config.train_annotation_folder,
                                                                  index, config.total_threads, min(max_images_per_thread, image_limit), train_mutex))
            image_limit -= max_images_per_thread
            thrd_list.append(thread)
            thread.start()

    image_limit = config.valid_dataset_generation_limit
    if image_limit > 0:
        max_images_per_thread = int(math.ceil(image_limit / config.total_threads))
        for index in range(config.total_threads):
            thread = threading.Thread(target=build_dataset, args=(config.valid_image_folder, config.valid_mask_folder, config.valid_annotation_folder,
                                                                  index, config.total_threads, min(max_images_per_thread, image_limit), valid_mutex))
            image_limit -= max_images_per_thread
            thrd_list.append(thread)
            thread.start()

    for thrd in thrd_list:
        thrd.join()

    # report the image with the smallest width, the image with the smallest height and the image with the smallest surface area
    print("Image with the smallest height (w, h): (%d, %d)" % (stats.minimum_height_img_w, stats.minimum_height_img_h))
    print("Image with the smallest width (w, h): (%d, %d)" % (stats.minimum_width_img_w, stats.minimum_width_img_h))
    print("Image with the smallest area (w, h): (%d, %d)" % (stats.minimum_area_img_w, stats.minimum_area_img_h))
