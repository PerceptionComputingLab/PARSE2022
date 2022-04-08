import numpy as np
from config import opt
import feature as feat
from tqdm import tqdm
from model import UNet3D
import torch
import dataset
from torch.utils.data import DataLoader
from datetime import datetime


def dice_coefficient(y_pred, y_true):
    smooth = 0.00001
    y_true_f, y_pred_f = y_true.flatten(), y_pred.flatten()
    intersection = np.logical_and(y_pred_f, y_true_f)
    return (2. * intersection.sum() + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def pred2seg(pred, target):
    if opt.use_gpu:
        pred, target = pred.cpu(), target.cpu()
    pred, target = pred.detach().numpy(), target.detach().numpy()
    pred[pred <= 0.5], pred[pred > 0.5] = 0, 1
    pred, target = pred.astype(np.uint8), target.astype(np.uint8)
    return pred, target

def evalu_dice(evalu_model, eval_dataset, eval_dataLoader, eval_total=1.0):

    dice_counter = feat.Counter()
    eval_total, tmp_total = int(eval_total * len(eval_dataset)), 0
    for batch_id, (dcm_image, label_image) in tqdm(enumerate(eval_dataLoader),
                                                   total=int(eval_total / opt.batch_size)):
        if (tmp_total + 1) * opt.batch_size > eval_total:
            break
        if dcm_image.shape[0] < opt.batch_size and eval_total < tmp_total:
            continue
        Input, target = dcm_image.requires_grad_(), label_image
        if opt.use_gpu is True:
            Input, target = Input.cuda(), target.cuda()
        pred = evalu_model(Input)
        pred, target = pred2seg(pred, target)
        dice = dice_coefficient(pred, target)
        dice_counter.updata(dice)
        tmp_total += 1
    return round(dice_counter.avg * 100, 1)

def train_dice_check():

    Unet = UNet3D(1, 1)
    Unet.load_state_dict(torch.load(opt.root_model_param))
    if opt.use_gpu is True :
        Unet = Unet.cuda()
    Unet.eval()
    train_dice_dataset = dataset.volume_dataset()
    train_dice_dataLoader = DataLoader(train_dice_dataset, batch_size=opt.batch_size, shuffle=False)
    print('train dice loading finish')

    print(evalu_dice(Unet, train_dice_dataset, train_dice_dataLoader))
    return

def pred_dcm_array():

    dcm_array = np.load(opt.root_pred_dcm)
    print(dcm_array.shape)

    dcm_volume = dcm_array.astype(np.float32)
    image_mean, image_std = np.mean(dcm_volume), np.std(dcm_volume)
    dcm_volume = (dcm_volume - image_mean) / image_std

    znum, xnum, ynum = int(dcm_volume.shape[0] / 92), int(dcm_volume.shape[1] / 92), int(dcm_volume.shape[2] / 92)
    if dcm_volume.shape[0] % 92:
        znum += 1
    if dcm_volume.shape[1] % 92:
        xnum += 1
    if dcm_volume.shape[2] % 92:
        ynum += 1

    print(znum*96, xnum*96, ynum*96)

    volume_array, pred_array = np.zeros((znum * xnum * ynum, 96, 96, 96), dtype=np.float32), np.zeros(
        (dcm_volume.shape[0], dcm_volume.shape[1], dcm_volume.shape[2]), dtype=np.uint8)
    for i in range(znum):
        for j in range(xnum):
            for k in range(ynum):
                tmp_array = dcm_volume[i * 96: min(dcm_volume.shape[0], (i + 1) * 96),
                                    j * 96: min(dcm_volume.shape[1], (j + 1) * 96),
                                    k * 96: min(dcm_volume.shape[2], (k + 1) * 96)]

                id = i * xnum * ynum + j * ynum + k
                volume_array[id, :tmp_array.shape[0], :tmp_array.shape[1], :tmp_array.shape[2]] = tmp_array[:, :, :]

    Unet = UNet3D(1, 1)
    Unet.load_state_dict(torch.load(opt.root_model_param))

    if opt.use_gpu is True:
        Unet = Unet.cuda()

    Unet.eval()

    for i in range(volume_array.shape[0]):

        tmp_volume = torch.tensor(volume_array[i], dtype=torch.float32)
        tmp_volume = torch.unsqueeze(torch.unsqueeze(tmp_volume, 0), 0)

        Input = tmp_volume.requires_grad_()
        if opt.use_gpu is True:
            Input = Input.cuda()

        pred = Unet(Input)

        if opt.use_gpu:
            pred = pred.cpu()

        pred = torch.squeeze(torch.squeeze(pred, 0), 0)

        pred = pred.detach().numpy()
        pred[pred <= 0.5], pred[pred > 0.5] = 0, 1
        pred = pred.astype(np.uint8)

        tznum = int(i / (xnum * ynum))
        txnum = int((i - tznum * (xnum * ynum)) / ynum)
        tynum = i - tznum * (xnum * ynum) - txnum * ynum

        pred_array[tznum * 96:min(dcm_volume.shape[0], (tznum + 1) * 96),
        txnum * 96:min(dcm_volume.shape[1], (txnum + 1) * 96),
        tynum * 96:min(dcm_volume.shape[2], (tynum + 1) * 96)] = pred[: min(dcm_volume.shape[0], (tznum + 1) * 96) - tznum * 96,
                                                                    : min(dcm_volume.shape[1], (txnum + 1) * 96) - txnum * 96,
                                                                    : min(dcm_volume.shape[2], (tynum + 1) * 96) - tynum * 96]

        time = datetime.now()
        np.save(
            f'{opt.root_pred_save}{str(time.month).zfill(2)}{str(time.day).zfill(2)}_{str(time.hour).zfill(2)}{str(time.minute).zfill(2)}.npy',pred_array)

    return