import os
import torch
import numpy as np
from torch.utils import data
import nibabel as nib
from config import opt
from tqdm import tqdm
import feature as feat


def folder_creation():

    if not os.path.exists(opt.root_dataset_file):
        os.makedirs(opt.root_dataset_file)
    if not os.path.exists(opt.root_train_volume):
        os.makedirs(opt.root_train_volume)

    return


def get_cup_extre(image):
    cup_extre = np.zeros([3, 2], dtype=np.int16)
    flag = False
    for i in range(image.shape[0]):
        tmp = np.sum(image[i, :, :])
        if tmp > 0:
            cup_extre[0, 1] = i
            if flag == False:
                flag = True
                cup_extre[0, 0] = i
    cup_extre[0, 0] = max(0, cup_extre[0, 0] - 5)
    cup_extre[0, 1] = min(image.shape[0], cup_extre[0, 1] + 6)
    flag = False
    for i in range(image.shape[1]):
        tmp = np.sum(image[:, i, :])
        if tmp > 0:
            cup_extre[1, 1] = i
            if flag == False:
                flag = True
                cup_extre[1, 0] = i
    cup_extre[1, 0] = max(0, cup_extre[1, 0] - 5)
    cup_extre[1, 1] = min(image.shape[1], cup_extre[1, 1] + 6)
    flag = False
    for i in range(image.shape[2]):
        tmp = np.sum(image[:, :, i])
        if tmp > 0:
            cup_extre[2, 1] = i
            if flag == False:
                flag = True
                cup_extre[2, 0] = i
    cup_extre[2, 0] = max(0, cup_extre[2, 0] - 5)
    cup_extre[2, 1] = min(image.shape[2], cup_extre[2, 1] + 6)
    return cup_extre


def rawdata_loading():
    root_rawfile = opt.root_raw_train_data
    root_volume_file = opt.root_train_volume
    fileList = os.listdir(root_rawfile)
    for f in tqdm(fileList, total=len(fileList)):
        tmp_dcmfile, tmp_labelfile = f'{root_rawfile}{f}/image/', f'{root_rawfile}{f}/label/'
        root_dcmfile, root_labelfile = f'{tmp_dcmfile}{os.listdir(tmp_dcmfile)[0]}', f'{tmp_labelfile}{os.listdir(tmp_labelfile)[0]}'
        
        dcm_img, label_img = np.array(nib.load(root_dcmfile).dataobj), np.array(nib.load(root_labelfile).dataobj)
        dcm_img, label_img = np.swapaxes(dcm_img, 0, 2), np.swapaxes(label_img, 0, 2)
        
        cup_extre = get_cup_extre(label_img)
        dcm_img = dcm_img[cup_extre[0, 0]:cup_extre[0, 1], cup_extre[1, 0]:cup_extre[1, 1],
                  cup_extre[2, 0]:cup_extre[2, 1]]
        label_img = label_img[cup_extre[0, 0]:cup_extre[0, 1], cup_extre[1, 0]:cup_extre[1, 1],
                    cup_extre[2, 0]:cup_extre[2, 1]]
        dcm_img = feat.add_window(dcm_img)
        label_img = label_img.astype(np.bool)
        root_save = f'{root_volume_file}{f}/'
        if not os.path.exists(root_save):
            os.makedirs(root_save)
        np.save(f'{root_save}/dcm.npy', dcm_img)
        np.save(f'{root_save}/label.npy', label_img)


def volume_resize():
    root_volumefile = opt.root_train_volume
    fileList = os.listdir(root_volumefile)
    
    dcm_array, label_array = np.zeros([100, 96 * 3, 96 * 3, 96 * 4], dtype=np.uint8), np.zeros(
        [100, 96 * 3, 96 * 3, 96 * 4], dtype=np.bool)

    tot_file = 0
    for f in tqdm(fileList, total=len(fileList)):
        tmp_dcmfile, tmp_labelfile = f'{root_volumefile}{f}/dcm.npy', f'{root_volumefile}{f}/label.npy'
        dcmfile, labelfile = np.load(tmp_dcmfile), np.load(tmp_labelfile)
        dcmfile = dcmfile[:min(dcmfile.shape[0], 96 * 3), :min(dcmfile.shape[1], 96 * 3),
                  :min(dcmfile.shape[2], 96 * 4)]
        labelfile = labelfile[:min(labelfile.shape[0], 96 * 3), :min(labelfile.shape[1], 96 * 3),
                    :min(labelfile.shape[2], 96 * 4)]

        dcmfile = np.pad(dcmfile, (
        (0, 96 * 3 - dcmfile.shape[0]), (0, 96 * 3 - dcmfile.shape[1]), (0, 96 * 4 - dcmfile.shape[2])),
                         mode='symmetric')
        labelfile = np.pad(labelfile, (
        (0, 96 * 3 - labelfile.shape[0]), (0, 96 * 3 - labelfile.shape[1]), (0, 96 * 4 - labelfile.shape[2])),
                         mode='symmetric')
        
        dcm_array[tot_file, :, :, :] = dcmfile[:, :, :]
        label_array[tot_file, :, :, :] = labelfile[:, :, :]
        
        tot_file += 1
    np.save(f'{opt.root_dataset_file}dcm_volume_array.npy', dcm_array)
    np.save(f'{opt.root_dataset_file}label_volume_array.npy', label_array)


def check_volume_array(check_id=0):
    root_volume = opt.root_dataset_file
    dcm_array = np.load(f'{root_volume}dcm_volume_array.npy')
    label_array = np.load(f'{root_volume}label_volume_array.npy')

    print(dcm_array.shape, label_array.shape)
    np.save(f'{root_volume}0D.npy', dcm_array[check_id])
    np.save(f'{root_volume}0L.npy', label_array[check_id])
    return


def dataset_preprocessing():
    folder_creation()
    print('folder creation finish!')
    rawdata_loading()
    print('rawdata loading finish!')
    volume_resize()
    print('volume creation finish!')
    return


class volume_dataset(data.Dataset):
    def __init__(self, range_data=None):
        if range_data is None:
            range_data = [0, 1]
        self.num_epoch = 0
        dcm_array, label_array = np.load('./dataset/dcm_volume_array.npy'), np.load('./dataset/label_volume_array.npy')
        self.len_volume = dcm_array.shape[0]

        self.load_dcm_array = dcm_array[int(self.len_volume * range_data[0]): int(self.len_volume * range_data[1])]
        self.load_label_array = label_array[int(self.len_volume * range_data[0]): int(self.len_volume * range_data[1])]
        return

    def update_num_epoch(self, num_epoch):
        self.num_epoch = num_epoch
        return

    def __getitem__(self, index):

        num_volume, num_patch = int(index / (3 * 3 * 4)), index % (3 * 3 * 4)
        dcm_volume, label_volume = self.load_dcm_array[num_volume], self.load_label_array[num_volume]
        dcm_volume = dcm_volume.astype(np.float32)
        image_mean, image_std = np.mean(dcm_volume), np.std(dcm_volume)
        dcm_volume = (dcm_volume - image_mean) / image_std
        
        numx = int(num_patch / 12)
        numy = int((num_patch - numx * 12) / 4)
        numz = num_patch - numx * 12 - numy * 4
        ret_dcm = dcm_volume[numx * 96:(numx + 1) * 96, numy * 96:(numy + 1) * 96, numz * 96:(numz + 1) * 96]
        ret_label = label_volume[numx * 96:(numx + 1) * 96, numy * 96:(numy + 1) * 96, numz * 96:(numz + 1) * 96]

        ret_dcm = torch.tensor(ret_dcm, dtype=torch.float32)
        ret_label = torch.tensor(ret_label, dtype=torch.long)
        ret_dcm, ret_label = torch.unsqueeze(ret_dcm, 0), torch.unsqueeze(ret_label, 0)

        return ret_dcm, ret_label

    def __len__(self):

        return self.load_dcm_array.shape[0] * 3 * 3 * 4
