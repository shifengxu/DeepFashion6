#!/usr/bin/env python
import os
import numpy as np
import cv2
import sys

CATEGORY_MATRIX = [
    ['floral', 'graphic', 'striped', 'embroidered', 'pleated', 'solid', 'lattice'],
    ['long_sleeve', 'short_sleeve', 'sleeveless'],
    ['maxi_length', 'mini_length', 'no_dress'],
    ['crew_neckline', 'v_neckline', 'square_neckline', 'no_neckline'],
    ['denim', 'chiffon', 'cotton', 'leather', 'faux', 'knit'],
    ['tight', 'loose', 'conventional']
]

def drawLandmarks(imageData, lmark):
    for i in range(8):
        left = lmark[i * 2 + 0]
        top = lmark[i * 2 + 1]
        if left == 0 and top == 0:
            continue
        right = left + 4
        bottom = top + 4
        cv2.rectangle(imageData, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(imageData, f"{i}", (left, top + 3), 0, 0.5, (0, 0, 255), 2)
    # for


def main_narrow():
    split_dir = "../data/split"
    data_dir  = "../data"
    data_type = "val"
    with open(os.path.join(split_dir, f"{data_type}.txt"), "r") as f:
        # train.txt has line like this: img/00000.jpg
        images = f.read().splitlines()
    lmarks = np.loadtxt(os.path.join(split_dir, f"{data_type}_landmards.txt"), dtype='i', delimiter=' ')
    bboxes = np.loadtxt(os.path.join(split_dir, f"{data_type}_bbox.txt"), dtype='i', delimiter=' ')
    dir = f"{data_dir}/tmp_narrow"
    if not os.path.exists(dir):
        os.makedirs(dir)
    for index in range(len(images)):
        img_path = os.path.join(data_dir, images[index])
        imgcv = cv2.imread(img_path)
        image_h, image_w, _ = imgcv.shape
        bbox = bboxes[index]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        # if h < 70:
        #     cv2.rectangle(imgcv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        #     _adjust_bbox_height(bbox, imgcv)
        #     l, t, r, b = bbox
        #     cv2.rectangle(imgcv, (l, t), (r, b), (255, 0, 0), 2)
        #     drawLandmarks(imgcv, lmarks[index])
        #     cv2.imwrite(f"{dir}/{os.path.split(img_path)[-1]}-{l}-{t}-{r}-{b}.jpg", imgcv)
        #     print(f"{img_path}")
        if w <= 70 and h > 100:
            cv2.rectangle(imgcv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            _adjust_bbox_width(bbox, imgcv)
            l, t, r, b = bbox
            cv2.rectangle(imgcv, (l, t), (r, b), (255, 0, 0), 2)
            drawLandmarks(imgcv, lmarks[index])
            cv2.imwrite(f"{dir}/{os.path.split(img_path)[-1]}-{l}-{t}-{r}-{b}.jpg", imgcv)
            print(f"{img_path}")
    # for


def _adjust_bbox_height(bbox, image, threshold=70):
    image_h, image_w, _ = image.shape
    l, t, r, b = bbox  # left, top, right, bottom
    b += 20
    bbox[3] = b


def _adjust_bbox_width(bbox, image, threshold=70):
    image_h, image_w, _ = image.shape
    l, t, r, b = bbox # left, top, right, bottom
    old_width = r - l
    old_height = b - t
    if old_height >= 200:
        new_width = 100 + (old_height - 200) // 2
    elif old_height >= 150:
        new_width = 90 + (old_height - 150) // 2
    elif old_height >= 100:
        new_width = threshold + (old_height - 100) // 2
    else:
        new_width = old_width
    if new_width < threshold: new_width = threshold
    if new_width <= old_width: return
    if new_width > old_height * 2: new_width = old_height * 2
    half = (new_width - old_width) // 2
    l -= half
    r += half
    if l < 0 : l = 0
    if r >= image_w: r = image_w - 1
    bbox[0] = l
    bbox[2] = r


def main_cats():
    split_dir = "../data/split"
    data_dir  = "../data"
    data_type = "val"
    with open(os.path.join(split_dir, f"{data_type}.txt"), "r") as f:
        # train.txt has line like this: img/00000.jpg
        images = f.read().splitlines()
    lmarks = np.loadtxt(os.path.join(split_dir, f"{data_type}_landmards.txt"), dtype='i', delimiter=' ')
    bboxes = np.loadtxt(os.path.join(split_dir, f"{data_type}_bbox.txt"), dtype='i', delimiter=' ')
    labels = np.loadtxt(os.path.join(split_dir, f"{data_type}_attr.txt"), dtype='i', delimiter=' ')
    dir_cats = f"{data_dir}/tmp_cats"
    if not os.path.exists(dir_cats):
        os.makedirs(dir_cats)
    for index in range(len(images)):
        if index != 692: continue
        img_path = os.path.join(data_dir, images[index])
        imgcv = cv2.imread(img_path)
        bbox = bboxes[index]
        cv2.rectangle(imgcv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        drawLandmarks(imgcv, lmarks[index])
        label_arr = labels[index]
        for j in range(len(label_arr)):
            dir = f"{dir_cats}/cat{j}_{label_arr[j]}"
            if not os.path.exists(dir):
                os.makedirs(dir)
            out_path = f"{dir}/{os.path.split(img_path)[-1]}"
            cv2.imwrite(out_path, imgcv)
        # for
        print(f"{img_path}")
    # for


def main_bbox():
    split_dir = "../Project/data/split"
    data_dir  = "../Project/data"
    data_type = "test"
    with open(os.path.join(split_dir, f"{data_type}.txt"), "r") as f:
        # train.txt has line like this: img/00000.jpg
        images = f.read().splitlines()
    bboxes = np.loadtxt(os.path.join(split_dir, f"{data_type}_bbox.txt"), dtype='i', delimiter=' ')
    for index in range(len(images)):
        img_path = os.path.join(data_dir, images[index])
        imgcv = cv2.imread(img_path)
        bbox = bboxes[index]
        cv2.rectangle(imgcv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
        out_path = f"{data_dir}/tmp_bbox/{os.path.split(img_path)[-1]}"
        cv2.imwrite(out_path, imgcv)
        print(f"{img_path} ==> {out_path}")
    # for


def main_compare():
    split_dir = "../data/split"
    data_dir  = "../data"
    data_type = "val"
    with open(os.path.join(split_dir, f"{data_type}.txt"), "r") as f:
        # train.txt has line like this: img/00000.jpg
        images = f.read().splitlines()
    labels = np.loadtxt(os.path.join(split_dir, f"{data_type}_attr.txt"), dtype='i', delimiter=' ')
    lmarks = np.loadtxt(os.path.join(split_dir, f"{data_type}_landmards.txt"), dtype='i', delimiter=' ')
    bboxes = np.loadtxt(os.path.join(split_dir, f"{data_type}_bbox.txt"), dtype='i', delimiter=' ')
    results = np.loadtxt(os.path.join(data_dir, f"vald_results_ex_0.8510.txt"), dtype='i', delimiter=' ')
    dir_compare = f"{data_dir}/tmp_compare_e5_0.8510"
    if not os.path.exists(dir_compare):
        os.makedirs(dir_compare)
    cm = CATEGORY_MATRIX
    for index in range(len(images)):
        img_path = os.path.join(data_dir, images[index])
        imgcv = cv2.imread(img_path)
        l, t, r, b = bboxes[index] # left, top, right, bottom
        label_arr = labels[index]
        result_arr = results[index]
        for j in range(len(label_arr)):
            dir = f"{dir_compare}/cat{j}"
            if not os.path.exists(dir):
                os.makedirs(dir)
            re, lb = result_arr[j], label_arr[j]
            if re == lb: continue
            cv2.rectangle(imgcv, (l, t), (r, b), (0, 0, 255), 2)
            drawLandmarks(imgcv, lmarks[index])
            lbstr = cm[j][lb]
            restr = cm[j][re]
            cv2.imwrite(f"{dir}/{os.path.split(img_path)[-1][:-4]}-rt-{lbstr}-wro-{restr}.jpg", imgcv)
        # for
        print(f"{img_path}")
    # for


if __name__ == "__main__":
    print(sys.argv)
    # main_narrow()
    main_compare()
