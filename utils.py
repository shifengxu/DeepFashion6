import argparse
import datetime
import os
import time
import shutil
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torchvision
from dataset import (DataSource, FashionDataset)
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint:", filename)
    torch.save(state, filename)

def load_checkpoint(filename, model):
    checkpoint = torch.load(filename)
    print("=> Loading checkpoint", filename)
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    data_dir,
    batch_size,
    train_transform,
    val_transform,
    train_data_ranges=[(0, 5000)],
    val_data_ranges=[(5000, 6000)],
    test_data_ranges=[(6000, 7000)],
    num_workers=4,
    pin_memory=True,
):
    data_src = DataSource(data_dir)
    train_ds = FashionDataset(
        data_src=data_src,
        data_usage='train',
        data_ranges=train_data_ranges,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers, # how many subprocesses to use for data loading. default 0.
        pin_memory=pin_memory, # the data loader will copy Tensors into CUDA pinned memory before return them.
        shuffle=True,
    )

    val_ds = FashionDataset(
        data_src=data_src,
        data_usage='val',
        data_ranges=val_data_ranges,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_ds = FashionDataset(
        data_src=data_src,
        data_usage='test',
        data_ranges=test_data_ranges,
        transform=val_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

def check_accuracy(loader, model, cat_ids, device="cuda"):
    numerator_sum = 0
    denominator_sum = 0
    numerator_arr = [0] * 6
    denominator_arr = [0] * 6
    model.eval() # set the module in evaluation mode. equivalent with model2.train(False)
    with torch.no_grad(): # context-manager that disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            denominator = x.shape[0]
            for i in cat_ids:
                pred = torch.argmax(preds[i], dim=1)
                numerator = torch.eq(y[:, i], pred).sum()
                numerator_arr[i] += numerator
                denominator_arr[i] += denominator
                numerator_sum += numerator
                denominator_sum += denominator
            # for
        # for
    # with
    rate_arr = []
    for i in range(6):
        r = numerator_arr[i] / denominator_arr[i] if denominator_arr[i] != 0 else 0.0
        rate_arr.append(r)
    rate_arr.append(numerator_sum/denominator_sum)
    str = rate_arr_to_str(rate_arr)
    print(f"===> Got {numerator_sum}/{denominator_sum} with accu {str}")
    model.train() # set the module in training mode
    return rate_arr

def rate_arr_to_str(rate_arr):
    str = f"{rate_arr[-1]:.4f}({rate_arr[0]:.4f}"
    for i in range(1, len(rate_arr) - 1):
        str = f"{str} {rate_arr[i]:.4f}"
    return f"{str})"

def convertArray(matrix, new_arr_len, device="cuda"):
    matrix_len = len(matrix)
    old_arr_len = len(matrix[0])
    res_matrix = torch.zeros([matrix_len, new_arr_len], dtype=torch.float, device=device)
    old_arr_seg = old_arr_len - 1
    new_arr_seg = new_arr_len - 1
    for ri in range(matrix_len): # first dimension is batch, such as 64
        old_arr = matrix[ri]
        res_matrix[ri][0] = old_arr[0]
        res_matrix[ri][-1] = old_arr[-1]
        for j in range(1, new_arr_len - 1):
            idx1 = old_arr_seg * j // new_arr_seg
            remn = old_arr_seg * j % new_arr_seg
            if remn == 0:
                res_matrix[ri][j] = old_arr[idx1]
            else:
                idx2 = idx1 + 1
                res_matrix[ri][j] = (old_arr[idx2] * remn + old_arr[idx1] * (new_arr_seg - remn)) / new_arr_seg
        res_matrix[ri]
    return res_matrix

# Calculate and save the prediction result, on validation or test data
def calc_save_prediction_result(loader, model, cat_ids, filename, device='cuda'):
    print("=> Calculate and save result to", filename)
    model.eval() # set the module in evaluation mode. equivalent with model2.train(False)
    col_list = [[], [], [], [], [], []]
    with torch.no_grad(): # context-manager that disable gradient calculation
        for x in loader:
            if isinstance(x, list):
                x = x[0] # validation data-set returns [imageTensor, labelTensor]
            x = x.to(device)
            preds = model(x)
            for i in cat_ids:
                pred = torch.argmax(preds[i], dim=1)
                col_list[i].extend(pred)
        # for
    model.train() # set the module in training mode
    row_cnt = len(col_list[0])
    with open(filename, 'w') as f:
        for i in range(row_cnt):
            f.write("%d %d %d %d %d %d\n" % (col_list[0][i], col_list[1][i], col_list[2][i],
                                             col_list[3][i], col_list[4][i], col_list[5][i]))


def calc_accuracies(fnames, label_matrix):
    if len(fnames) != 6: raise Exception(f"fnames len ({len(fnames)}) is not 6.")
    matrix0 = []
    for fname in fnames:
        m = np.loadtxt(fname, dtype='i', delimiter=' ')
        matrix0.append(m)
    # for
    # now, compare matrix0 and label_matrix
    numerator_sum = 0
    denominator_sum = 0
    rate_arr = []
    denominator = len(matrix0[0])
    label_matrix = np.asarray(label_matrix).transpose()
    for i in range(6):
        numerator = torch.eq(torch.tensor(matrix0[i]), torch.tensor(label_matrix[i])).sum()
        numerator_sum += numerator
        denominator_sum += denominator
        rate_arr.append(numerator/denominator)
    rate_arr.append(numerator_sum / denominator_sum)
    return rate_arr


# Calculate and save the prediction result, on validation or test data
def save_prediction_result_cat(cat_id, loader, model, filename, device='cuda'):
    print("=> Saving prediction:", filename)
    model.eval() # set the module in evaluation mode. equivalent with model2.train(False)
    row_list = []
    with torch.no_grad(): # context-manager that disable gradient calculation
        for x in loader:
            if isinstance(x, list):
                x = x[0] # validation data-set returns [imageTensor, labelTensor]
            x = x.to(device)
            preds = model(x)
            pred = torch.argmax(preds[cat_id], dim=1)
            row_list.extend(pred)
        # for
    model.train() # set the module in training mode
    with open(filename, 'w') as f:
        for i in row_list:
            f.write("%d\n" % i)

def float_arr_to_str(f_arr):
    return '[{:s}]'.format(' '.join(['{:-.4f}'.format(x) for x in f_arr]))

def dtstr():
    now = time.time() + 60 * 60 * 8  # seconds. change timezone to UTC+08:00
    return datetime.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def zip_result_file(src_filename, zipped_filename, unzipped_filename=None):
    print(f"=> create file {zipped_filename}")
    if unzipped_filename:
        shutil.copyfile(src_filename, unzipped_filename)
    else:
        unzipped_filename = src_filename
    with zipfile.ZipFile(zipped_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(unzipped_filename)


def zip_result_files(src_fnames, zipped_fname, unzipped_fname):
    arr_arr = []
    for fname in src_fnames:
        with open(fname, 'r') as f:
            arr = f.read().splitlines()
            arr_arr.append(arr)
    r_cnt = len(arr_arr)
    c_cnt = len(arr_arr[0])
    with open(unzipped_fname, 'w') as f:
        for c in range(c_cnt):
            for r in range(r_cnt - 1):
                f.write(f"{arr_arr[r][c]} ")
            f.write(f"{arr_arr[-1][c]}\n")
    with zipfile.ZipFile(zipped_fname, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(unzipped_fname)
