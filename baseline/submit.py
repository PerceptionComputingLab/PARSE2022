import numpy as np
from config import opt
import feature as feat
from tqdm import tqdm
from model import UNet3D
import torch
import dataset
from torch.utils.data import DataLoader
from datetime import datetime
import nibabel as nib
import os
import SimpleITK as sitk
import evalu


def eval_data_maker(label_mask=False, lung_mask=False):

    root_eval_data = opt.root_raw_eval_data
    root_eval_volume = opt.root_eval_volume

    if not os.path.exists(root_eval_volume):
        os.makedirs(root_eval_volume)

    fileList = os.listdir(root_eval_data)
    for i in tqdm(range(len(fileList))):
        root_data = f'{root_eval_data}{fileList[i]}/'
        root_dcm = f'{root_data}image/{fileList[i]}.nii.gz'

        dcm_img = np.swapaxes(np.array(nib.load(root_dcm).dataobj), 0, 2)
        dcm_img = feat.add_window(dcm_img)

        root_save = f'{root_eval_volume}{fileList[i]}/'
        if not os.path.exists(root_save):
            os.makedirs(root_save)
        np.save(f'{root_save}/dcm.npy', dcm_img)

        # feat.info_data(dcm_img)
        # feat.info_data(label_img)
        if label_mask:
            root_label = f'{root_data}label/{fileList[i]}.nii.gz'
            label_img = np.swapaxes(np.array(nib.load(root_label).dataobj), 0, 2)
            label_img = label_img.astype(np.uint8)
            np.save(f'{root_save}/label.npy', label_img)

        if lung_mask:
            root_lung = f'{root_data}lung/{fileList[i]}.nii.gz'
            lung_img = np.swapaxes(np.array(nib.load(root_lung).dataobj), 0, 2)
            np.save(f'{root_save}/lung.npy', lung_img)
            feat.info_data(lung_img)


def pred_folder_creation():

    if not os.path.exists(opt.root_eval_volume):
        os.makedirs(opt.root_eval_volume)

    root_submit_file = opt.root_submit_file
    if not os.path.exists(opt.root_submit_file):
        os.makedirs(opt.root_submit_file)
    if not os.path.exists(f'{root_submit_file}npy/'):
        os.makedirs(f'{root_submit_file}npy/')
    if not os.path.exists(f'{root_submit_file}nii/'):
        os.makedirs(f'{root_submit_file}nii/')


def eval_data(eval_model, save_pred=False, root_save=None, dice_calcu=False):

    dice_counter=None
    if dice_calcu:
        dice_counter = feat.Counter()

    root_eval = opt.root_eval_volume
    fileList = os.listdir(root_eval)

    for i in tqdm(range(len(fileList))):

        root_data = f'{root_eval}{fileList[i]}/'
        root_dcm = f'{root_data}dcm.npy'
        array_dcm = np.load(root_dcm)
        array_pred = pred_volume(eval_model, array_dcm)
        # print(array_pred.shape, array_label.shape)
        if dice_calcu:
            root_label = f'{root_data}label.npy'
            array_label = np.load(root_label)
            dice = evalu.dice_coefficient(array_pred, array_label)
            dice_counter.updata(dice)

        if save_pred:
            np.save(f'{root_save}{fileList[i]}.npy', array_pred)

    if dice_calcu:
        print(f"prediction averaged dice:", round(dice_counter.avg * 100, 1))
    return



def pred_volume(pred_model, dcm_array):

    dcm_volume = dcm_array.astype(np.float32)
    image_mean, image_std = np.mean(dcm_volume), np.std(dcm_volume)
    dcm_volume = (dcm_volume - image_mean) / image_std

    znum, xnum, ynum = int(dcm_volume.shape[0] / 96), int(dcm_volume.shape[1] / 96), int(dcm_volume.shape[2] / 96)
    if dcm_volume.shape[0] % 96:
        znum += 1
    if dcm_volume.shape[1] % 96:
        xnum += 1
    if dcm_volume.shape[2] % 96:
        ynum += 1

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

    if opt.use_gpu is True:
        pred_model = pred_model.cuda()

    pred_model.eval()

    for i in range(volume_array.shape[0]):

        tmp_volume = torch.tensor(volume_array[i], dtype=torch.float32)
        tmp_volume = torch.unsqueeze(torch.unsqueeze(tmp_volume, 0), 0)

        Input = tmp_volume.requires_grad_()
        if opt.use_gpu is True:
            Input = Input.cuda()

        pred = pred_model(Input)

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

    return pred_array


def numpy2niigz(root_numpy, root_niigz):
    fileList = os.listdir(root_numpy)
    for i in range(len(fileList)):
        root_data = f'{root_numpy}{fileList[i]}'
        out = sitk.GetImageFromArray(np.load(root_data))
        file_name = str(fileList[i]).split('.')[0]
        sitk.WriteImage(out, f'{root_niigz}/{file_name}.nii.gz')


def submit_pred(dice_calcu=False):

    pred_folder_creation()
    print('folder creation finish!')
    eval_data_maker(label_mask=dice_calcu)

    root_submit_file = opt.root_submit_file
    Unet = UNet3D(1, 1)
    Unet.load_state_dict(torch.load(opt.root_model_param))
    print('model loading finish!')

    pred_numpy_array = f'{root_submit_file}npy/'
    pred_nii_array = f'{root_submit_file}nii/'

    eval_data(Unet, save_pred=True, root_save=pred_numpy_array, dice_calcu=dice_calcu)
    print('volume prediction finish!')
    numpy2niigz(pred_numpy_array, pred_nii_array)
    print('niigz changing finish!')

    return