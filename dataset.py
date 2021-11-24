import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A


class DataSource:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.bboxes = []
        self.lmarks = []
        self.labels = []
        split_dir = os.path.join(data_dir, 'split')
        for du in ['train', 'val', 'test']:  # for each data-usage
            fpath = os.path.join(data_dir, 'split', f"{du}_bbox.txt")
            self.bboxes.extend(np.loadtxt(fpath, dtype='i', delimiter=' '))
            fpath = os.path.join(data_dir, 'split', f"{du}_landmards.txt")
            self.lmarks.extend(np.loadtxt(fpath, dtype='i', delimiter=' '))
        # for du in ['train', 'val']:
            fpath = os.path.join(data_dir, 'split', f"{du}_attr.txt")
            self.labels.extend(np.loadtxt(fpath, dtype='i', delimiter=' '))
        print(f"---------- datasource() ----------")
        print(f"  data_dir  : {data_dir}")
        print(f"  labels len: {len(self.labels)}")
        print(f"  bboxes len: {len(self.bboxes)}")
        print(f"  lmarks len: {len(self.lmarks)}")

    def get_image(self, index):
        img_path = os.path.join(self.data_dir, 'img', f"{index:05}.jpg")
        image = np.array(Image.open(img_path).convert("RGB"))
        return image

    def get_bbox(self, index):
        return self.bboxes[index]

    def get_lmark(self, index):
        return self.lmarks[index]

    def get_label(self, index):
        return self.labels[index]
# class DataSource


class FashionDataset(Dataset):
    def __init__(self, data_src, data_usage, data_ranges, transform=None):
        self.data_src = data_src
        self.data_usage = data_usage
        self.transform = transform
        self.lmark_radius = 10
        self.lmark_ratio = 2
        self.adjust_bbox = True
        self.keep_hw_ratio = True
        self.img_idx_list = []
        for dr in data_ranges:  # each data-range has 2 elements
            for i in range(dr[0], dr[1]):
                self.img_idx_list.append(i)
        print(f"---------- dataset({data_usage}) ----------")
        print(f"  data_ranges  : {data_ranges}")
        print(f"  img_idx_list : {len(self.img_idx_list)} elements")
        print(f"  img_idx[0]   : {self.img_idx_list[0]}")
        print(f"  img_idx[-1]  : {self.img_idx_list[-1]}")
        print(f"  lmark_radius : {self.lmark_radius}")
        print(f"  lmark_ratio  : {self.lmark_ratio}")
        print(f"  adjust_bbox  : {self.adjust_bbox}")
        print(f"  keep_hw_ratio: {self.keep_hw_ratio}")

    def __len__(self):
        return len(self.img_idx_list)

    def __getitem__(self, index):
        img_idx = self.img_idx_list[index]
        image = self.data_src.get_image(img_idx)

        # image = self.tf_nml(image=image)["image"]  # -------------------------- normalize

        # self._apply_landmark(image, self.lmarks[index])

        bbox = self.data_src.get_bbox(img_idx)
        if self.adjust_bbox:
            self._adjust_bbox(bbox, image)
        image = image[bbox[1]:bbox[3]:, bbox[0]:bbox[2]:]  # ------------------ apply bbox
        if self.keep_hw_ratio:
            image = self._keep_hw_ratio(image)

        # image = self._append_landmark_layer(image, self.lmarks[index])

        # if self.lmarks[index].any(): # some element not 0
        #     augmentations = self.tf_196(image=image)
        #     image = augmentations["image"]  # --------------------------------- resize
        #     image = self._apply_landmark2(image, image0, self.lmarks[index])
        # else: # all zeros
        #     augmentations = self.tf_224(image=image)
        #     image = augmentations["image"]  # --------------------------------- resize

        # im = Image.fromarray(image) # for test only
        # im.save(f"./tmp/{index:05d}.jpg")

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        label = self.data_src.get_label(img_idx)
        return image, torch.tensor(label, dtype=torch.long)
    # __getitem__()

    # make a square image, but keep height-width ratio
    def _keep_hw_ratio(self, image):
        old_h, old_w, old_d = image.shape  # height, width, depth
        if old_h == old_w:
            return image
        if old_h > old_w:
            new_image = np.zeros((old_h, old_h, old_d), dtype=np.uint8)
            delta = old_h - old_w
            w1 = delta // 2
            w2 = w1 + old_w
            new_image[:, :w1, :] = 255  # white color
            new_image[:, w1:w2, :] = image
            new_image[:, w2:, :] = 255
        else:
            new_image = np.zeros((old_w, old_w, old_d), dtype=np.uint8)
            delta = old_w - old_h
            h1 = delta // 2
            h2 = h1 + old_h
            new_image[:h1, :, :] = 255  # white color
            new_image[h1:h2, :, :] = image
            new_image[h2:, :, :] = 255
        return new_image

    def _adjust_bbox(self, bbox, image, threshold=70):
        image_h, image_w, _ = image.shape
        l, t, r, b = bbox # left, top, right, bottom
        if r - l > 50: return 0
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
        return 1

    def _append_landmark_layer(self, image, mark_arr):
        hLen, wLen, dLen = image.shape  # height, width, depth
        new_layer = np.zeros((hLen, wLen, 1), dtype=np.uint8)
        radius = self.lmark_radius
        for i in range(8):
            wIdx = mark_arr[i * 2]      # width index
            hIdx = mark_arr[i * 2 + 1]  # height index
            if wIdx == 0 and hIdx == 0:
                continue
            w0 = 0 if wIdx <= radius else wIdx - radius
            h0 = 0 if hIdx <= radius else hIdx - radius
            w1 = wLen if wIdx + radius >= wLen else wIdx + radius
            h1 = hLen if hIdx + radius >= hLen else hIdx + radius
            new_layer[h0:h1, w0:w1, :] = 255
        return np.append(image, new_layer, axis=2)

    # put landmark widgets on the right part of the image
    def _apply_landmark2(self, image, image0, mark_arr):
        hLen, wLen, dLen = image.shape  # height, width, depth
        delta = 28
        new_image = np.zeros((hLen, wLen + delta, dLen), dtype=np.uint8)
        new_image.fill(255)
        new_image[:, 0:wLen, :] = image
        radius = delta // 2
        hLen0, wLen0, _ = image0.shape  # height, width, depth
        for i in range(8):
            wIdx = mark_arr[i * 2]      # width index
            hIdx = mark_arr[i * 2 + 1]  # height index
            if wIdx == 0 and hIdx == 0:
                continue
            w0 = 0 if wIdx <= radius else wIdx - radius
            h0 = 0 if hIdx <= radius else hIdx - radius
            w1 = wLen0 if w0 + delta >= wLen0 else w0 + delta
            h1 = hLen0 if h0 + delta >= hLen0 else h0 + delta
            new_image[i*28:i*28 + h1 - h0, wLen:wLen + w1 - w0, :] = image0[h0:h1, w0:w1, :]
        # for
        return new_image

    # change the pixel value, who is in the landmark radius: double the value
    def _apply_landmark(self, image, mark_arr):
        hLen, wLen, dLen = image.shape # height, width, depth
        matrix = np.zeros((hLen, wLen, dLen))
        radius = self.lmark_radius
        for i in range(8):
            wIdx = mark_arr[i * 2]      # width index
            hIdx = mark_arr[i * 2 + 1]  # height index
            if wIdx == 0 and hIdx == 0:
                continue
            w0 = 0 if wIdx <= radius else wIdx - radius
            h0 = 0 if hIdx <= radius else hIdx - radius
            w1 = wLen if wIdx + radius >= wLen else wIdx + radius
            h1 = hLen if hIdx + radius >= hLen else hIdx + radius
            matrix[h0:h1, w0:w1, :] = 0.5
        # for
        image *= 0.5
        image += matrix
    # _apply_landmark()

    def get_labels(self):
        res = []
        for i in self.img_idx_list:
            res.append(self.data_src.get_label(i))
        return res
# class

def test():
    category_idx = 2
    ds = FashionDataset("./data", 'train', category_idx)
    assert len(ds) == 5000
    assert len(ds.labels) == 5000
    assert ds.labels.shape == (5000,)
    assert ds.labels[0:10].tolist() == [2, 2, 2, 1, 1, 2, 2, 0, 2, 2]

    ds = FashionDataset("./data", 'val', category_idx)
    assert len(ds) == 1000
    assert len(ds.labels) == 1000
    assert ds.labels.shape == (1000,)
    assert ds.labels[0:10].tolist() == [2, 2, 0, 2, 1, 2, 2, 2, 2, 2]


if __name__ == '__main__':
    test()
