import math
import os
import threading

import numpy as np
import config
import random
from PIL import Image, ImageEnhance, ImageDraw, ImageMath


def build_dataset(thread_id, total_threads, limit, mutex):
    bkg_image_paths = [config.background_folder + x for x in os.listdir(config.background_folder)]
    labels_to_images = {}

    mutex.acquire()
    try:
        if not os.path.exists(config.image_folder):
            os.makedirs(config.image_folder)
        if not os.path.exists(config.mask_folder):
            os.makedirs(config.mask_folder)
    finally:
        mutex.release()

    for i in range(1, len(config.labels)):
        label = config.labels[i]
        img_paths = [config.dataset_train_folder + label + '/' + x for x in os.listdir(config.dataset_train_folder + label)]
        labels_to_images[label] = img_paths

    for index in range(limit):
        img_count = index * total_threads + thread_id
        canvas = np.array(Image.new(mode='RGB', size=config.img_size[:-1], color=(255, 255, 255)))
        mask_canvas = np.array(Image.new(mode='RGB', size=config.img_size[:-1], color=(0, 0, 0)))
        other_images = []
        background_image = np.array(Image.open(bkg_image_paths[random.randint(0, len(bkg_image_paths) - 1)]).resize(config.img_size[:-1]))
        fruits_in_image = random.randint(3, 5)
        for i in range(fruits_in_image):
            fruit_label_index = random.randint(1, len(config.labels) - 1)
            fruit_label = config.labels[fruit_label_index]
            fruit_mask_color = config.color_map[fruit_label_index]
            fruit_image_path = labels_to_images[fruit_label][random.randint(0, len(labels_to_images[fruit_label]) - 1)]
            fruit_img_size = random.randint(config.min_fruit_size, config.max_fruit_size)
            rotate_angle = random.randint(0, 3) * 90
            fruit_image = Image.open(fruit_image_path).resize((fruit_img_size, fruit_img_size)).rotate(rotate_angle)
            fruit_mask = build_mask(fruit_image, fruit_mask_color)
            fruit_image = np.array(fruit_image)
            add_image_and_mask_to_canvas(canvas, fruit_image, mask_canvas, fruit_mask, other_images)
        for i in range(canvas.shape[0]):
            for j in range(canvas.shape[1]):
                if (mask_canvas[i][j] == 0).all():
                    canvas[i][j] = background_image[i][j]
        canvas = Image.fromarray(canvas)
        canvas = enhance_image(canvas)
        mask_canvas = Image.fromarray(mask_canvas)
        canvas.save(config.image_folder + str(img_count) + '.png')
        mask_canvas.save(config.mask_folder + str(img_count) + '.png')


def enhance_image(canvas, sharpness=True, contrast=True, color=True, brightness=True):
    if sharpness:
        sharpness_enhancer = ImageEnhance.Sharpness(canvas)
        factor = random.random() * 0.6 + 0.7
        canvas = sharpness_enhancer.enhance(factor=factor)

    if contrast:
        contrast_enhancer = ImageEnhance.Contrast(canvas)
        factor = random.random() * 1.1 + 0.7
        canvas = contrast_enhancer.enhance(factor=factor)

    if color:
        color_enhancer = ImageEnhance.Color(canvas)
        factor = random.random() * 1.0 + 0.5
        canvas = color_enhancer.enhance(factor=factor)

    if brightness:
        brightness_enhancer = ImageEnhance.Brightness(canvas)
        factor = random.random() * 1.0 + 0.5
        canvas = brightness_enhancer.enhance(factor=factor)
    return canvas


def add_image_and_mask_to_canvas(canvas, fruit_image, canvas_mask, fruit_mask, other_images):
    done = False
    attempts = 10
    max_x = canvas.shape[0] - fruit_image.shape[0] - 1
    max_y = canvas.shape[1] - fruit_image.shape[1] - 1
    x = 0
    y = 0
    while not done and attempts > 0:
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        done = not is_overlap_between_new_image_and_old_images(((x, y), (x, y, fruit_image.shape[1]), (x + fruit_image.shape[0], y), (x + fruit_image.shape[0], y + fruit_image.shape[1])), other_images)
        attempts -= 1
    if done:
        for i in range(fruit_image.shape[0]):
            for j in range(fruit_image.shape[1]):
                canvas[x + i][y + j] = fruit_image[i][j]
                canvas_mask[x + i][y + j] = fruit_mask[i][j]
        other_images.append(((x, y), (x, y + fruit_image.shape[1]), (x + fruit_image.shape[0], y), (x + fruit_image.shape[0], y + fruit_image.shape[1])))
    return done


def is_overlap_between_new_image_and_old_images(img_coordinates, other_images):
    for old_img_coords in other_images:
        if is_src_img_inside_dest_img(img_coordinates, old_img_coords) or is_src_img_inside_dest_img(old_img_coords, img_coordinates):
            return True
    return False


def is_src_img_inside_dest_img(src_img_coordinates, dest_img_coordinates):
    upper_left_point = src_img_coordinates[0]
    upper_right_point = src_img_coordinates[1]
    lower_left_point = src_img_coordinates[2]
    lower_right_point = src_img_coordinates[3]
    height_upper_bound = dest_img_coordinates[0][0]  # as the upper left corner of an image is (0, 0), the upper bound will be smaller than the lower bound
    height_lower_bound = dest_img_coordinates[3][0]
    width_left_bound = dest_img_coordinates[0][1]
    width_right_bound = dest_img_coordinates[3][1]
    return ((height_upper_bound <= upper_left_point[0] <= height_lower_bound and width_left_bound <= upper_left_point[1] <= width_right_bound) or
            (height_upper_bound <= upper_right_point[0] <= height_lower_bound and width_left_bound <= upper_right_point[1] <= width_right_bound) or
            (height_upper_bound <= lower_left_point[0] <= height_lower_bound and width_left_bound <= lower_left_point[1] <= width_right_bound) or
            (height_upper_bound <= lower_right_point[0] <= height_lower_bound and width_left_bound <= lower_right_point[1] <= width_right_bound))


def build_mask(fruit_image, fruit_mask_color, threshold=config.mask_threshold):
    fn = lambda x: 0 if x > threshold else 255
    inv_fn = lambda x: 0 if x == 255 else 255
    img = fruit_image.convert('L').point(fn, mode='1')
    img_copy = img.copy()
    ImageDraw.floodfill(img_copy, (0, 0), 255)
    img_copy = img_copy.point(inv_fn, mode='1')
    img = ImageMath.eval("a | b", a=img, b=img_copy)
    img = img.convert('RGB')
    img_data = np.array(img)
    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            if (img_data[i][j] == 255).all():
                img_data[i][j] = fruit_mask_color

    return img_data


if __name__ == "__main__":
    thrd_list = []
    mutex = threading.Lock()
    image_limit = config.dataset_generation_limit
    max_images_per_thread = int(math.ceil(image_limit / config.total_threads))
    for index in range(config.total_threads):
        thread = threading.Thread(target=build_dataset, args=(index, config.total_threads, min(max_images_per_thread, image_limit), mutex))
        image_limit -= max_images_per_thread
        thrd_list.append(thread)
        thread.start()

    for thrd in thrd_list:
        thrd.join()
