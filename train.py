import argparse
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from model.resnet import (resnet152)
from loss.CrossEntropyLoss import CrossEntropyLoss
from loss.FocalLoss import FocalLoss
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    calc_accuracies,
    calc_save_prediction_result,
    save_prediction_result_cat,
    dtstr,
    str2bool,
    zip_result_file,
    zip_result_files,
    rate_arr_to_str,
)


CATEGORY_MATRIX = [
    ['floral', 'graphic', 'striped', 'embroidered', 'pleated', 'solid', 'lattice'],
    ['long_sleeve', 'short_sleeve', 'sleeveless'],
    ['maxi_length', 'mini_length', 'no_dress'],
    ['crew_neckline', 'v_neckline', 'square_neckline', 'no_neckline'],
    ['denim', 'chiffon', 'cotton', 'leather', 'faux', 'knit'],
    ['tight', 'loose', 'conventional']
]


def train_fn(args, loader, model, optimizer, loss_fn, epoch=0):
    # loop = tqdm(loader)
    loop = loader

    cat_ids = args.cat_ids # category id list
    loss_cnt = len(cat_ids)
    torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data, labels) in enumerate(loop):
        # data shape is [16, 3, 160, 240], and target shape is [16, 160, 240]. So need to unsqueeze target.
        data = data.to(device=args.device)
        labels = labels.to(device=args.device)

        predictions = model(data)
        loss_sum = 0.0
        optimizer.zero_grad()
        for i in cat_ids:
            loss = loss_fn(predictions[i], labels[:, i])
            loss_sum = loss_sum + loss.item()
            retain_graph = i != cat_ids[-1] # if not last id, then retain graph.
            loss.backward(retain_graph=retain_graph)
        optimizer.step()

        # update tqdm loop
        # loop.set_postfix(loss2=loss2.item())
        loss_avg = loss_sum / loss_cnt
        if batch_idx % 5 == 0:
            print(f"E({epoch:03d}/{args.num_epochs}).B({batch_idx:02d}) loss:{loss_avg:0.6f}")
    return loss_avg


def main(args):
    # Albumentations ensures that the input image and the output mask will
    # receive the same set of augmentations with the same parameters.
    TRAIN_TRANSFORM = A.Compose([  # define a list of augmentations
        A.Resize(height=args.image_height, width=args.image_width),
        A.Rotate(limit=35, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        # A.Equalize(p=1),
        A.Normalize(  # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),  # it is a class. To convert image and mask to torch.Tensor
    ])
    VAL_TRANSFORM = A.Compose([
        A.Resize(height=args.image_height, width=args.image_width),
        # A.Equalize(p=1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    train_loader, vald_loader, test_loader = get_loaders(
        args.data_dir,
        args.batch_size,
        TRAIN_TRANSFORM,
        VAL_TRANSFORM,
        train_data_ranges=[(0, 6000)],
        val_data_ranges=[(5000, 6000)],
        test_data_ranges=[(6000, 7000)],
        num_workers=2,
        pin_memory=True,
    )
    model = resnet152(pretrained=args.pretrained, cat_ids=args.cat_ids,
                      fork_layer34=args.fork_layer34, fork_layer4=args.fork_layer4)
    print(f"train.model: resnet152(pretrained={args.pretrained})")
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model.to(args.device)
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = CrossEntropyLoss()
    loss_fn = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"train.loss_fn: {type(loss_fn)}")
    print(f"train.optimizer: Adam(lr={args.learning_rate}, weight_decay={args.weight_decay})")
    # model2.parameters(): returns an iterator over module parameters. Typically passed to an optimizer

    if args.load_model:
        load_checkpoint(f"my_checkpoint.pth.tar", model)
    accu_arr = check_accuracy(vald_loader, model, args.cat_ids, device=args.device)
    if args.each_mode:
        accu_max_arr = [0.0] * 7
        save_cp_pred_each('x', accu_arr, accu_max_arr, model, test_loader, vald_loader)
    else:
        accu_max = accu_arr[-1]
        save_checkpoint_result('x', accu_max, model, test_loader, vald_loader)

    hist_file = f"hist.log"
    with open(hist_file, 'w') as f:
        f.write(f"{dtstr()} args: {args}\n")
    for e in range(args.num_epochs):
        loss = train_fn(args, train_loader, model, optimizer, loss_fn, e)
        accu_arr = check_accuracy(vald_loader, model, args.cat_ids, device=args.device)
        if args.each_mode:
            save_cp_pred_each(e, accu_arr, accu_max_arr, model, test_loader, vald_loader)
        elif accu_max < accu_arr[-1]:
            accu_max = accu_arr[-1]
            save_checkpoint_result(e, accu_max, model, test_loader, vald_loader)
        if e == args.num_epochs - 1:
            save_checkpoint({"state_dict": model.state_dict()}, f"my_checkpoint_end.pth.tar")
        with open(hist_file, 'a') as f:
            str = rate_arr_to_str(accu_arr)
            f.write(f"{dtstr()} E({e:03d}/{args.num_epochs}) loss:{loss:0.6f} accu:{str}\n")
    # for
    vald_fnames = [f"vald_results_cat{i}_max.txt" for i in range(6)]
    test_fnames = [f"test_results_cat{i}_max.txt" for i in range(6)]
    print("----- Calculate validation accuracy -----")
    [print(f"  {f}") for f in vald_fnames]
    accu_arr = calc_accuracies(vald_fnames, vald_loader.dataset.get_labels())
    print(f"{rate_arr_to_str(accu_arr)}")

    print("----- Zip test result files -----")
    zip_fname = f"prediction_{accu_arr[-1]:0.4f}.zip"
    [print(f"  {f}") for f in test_fnames]
    zip_result_files(test_fnames, zip_fname, "prediction.txt")
    print(zip_fname)


# save checkpoint, predictions in each mode
def save_cp_pred_each(epoch, accu_arr, accu_max_arr, model, test_loader, vald_loader):
    for i in range(6):
        if accu_max_arr[i] >= accu_arr[i]: continue
        accu_max_arr[i] = accu_arr[i]
        save_cp_pred_each_cat(i, epoch, accu_max_arr[i], model, test_loader, vald_loader)

    sum = 0.0
    for i in range(6):
        sum += accu_max_arr[i]
    accu_max_arr[6] = sum / 6
    print(f"now accu: {rate_arr_to_str(accu_arr)}")
    print(f"max accu: {rate_arr_to_str(accu_max_arr)}")


# save checkpoint and predictions
def save_cp_pred_each_cat(cid, epoch, accu, model, test_loader, vald_loader):
    if accu < 0.75: return
    print(f"Save file for: cat_id:{cid}, epoch:{epoch}, accu:{accu:0.4f}")
    save_prediction_result_cat(cid, test_loader, model, f"test_results_cat{cid}_max.txt", device=args.device)
    save_prediction_result_cat(cid, vald_loader, model, f"vald_results_cat{cid}_max.txt", device=args.device)
    save_checkpoint({"state_dict": model.state_dict()}, f"mycheckpoint_cat{cid}_max.pth.tar")


def save_checkpoint_result(epoch, accu, model, test_loader, vald_loader):
    if  accu < 0.84: return
    tr_fname = f"test_results_e{epoch}_{accu:0.4f}.txt"
    vr_fname = f"vald_results_e{epoch}_{accu:0.4f}.txt"
    pred_zip = f"prediction_e{epoch}_{accu:0.4f}.zip"
    cp_filename = f"my_checkpoint_e{epoch}_{accu:0.4f}.pth.tar"
    save_checkpoint({"state_dict": model.state_dict()}, cp_filename)
    calc_save_prediction_result(test_loader, model, args.cat_ids, tr_fname, device=args.device)
    calc_save_prediction_result(vald_loader, model, args.cat_ids, vr_fname, device=args.device)
    zip_result_file(tr_fname, pred_zip, "prediction.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CE7454 Fashion Attribute Assignment of shifeng001 (G2104007A)")
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[2, 3], help="GPU ID list")
    parser.add_argument('--cat-ids', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5], help="Category ID list")
    parser.add_argument('--num-epochs', type=int, default=0, help="number of epochs to run")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--image-height', type=int, default=224)
    parser.add_argument('--image-width', type=int, default=224)
    parser.add_argument('--load-model', type=str2bool, default=False)
    parser.add_argument('--pretrained', type=str2bool, default=True)
    parser.add_argument('--data-dir', type=str, default="../CE7454-A1-Proj/data")
    parser.add_argument('--model', type=str, default="resnet152", help="alexnet resnet50 resnet152, vgg19_bn")
    parser.add_argument('--fork-layer34', type=str2bool, default=False)
    parser.add_argument('--fork-layer4', type=str2bool, default=False)
    parser.add_argument('--each-mode', type=str2bool, default=True)
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else "cpu"
    print('args', args)

    main(args)
