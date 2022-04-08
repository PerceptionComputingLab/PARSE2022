import torch
import numpy as np
import os
import pandas as pd

def info_data(matrix_data):
    type_data = type(matrix_data)
    if type(matrix_data) is torch.Tensor:
        if matrix_data.is_cuda:
            matrix_data = matrix_data.detach().cpu()
        matrix_data = np.array(matrix_data)
    print('data_type/dtype/shape: ', type_data, matrix_data.dtype, matrix_data.shape)
    print('min/max: ', np.min(matrix_data), np.max(matrix_data), 'num</=/>zero: ', np.sum(matrix_data < 0),
          np.sum(matrix_data == 0), np.sum(matrix_data > 0))
    return

def create_root(root_name):
    if not os.path.exists(root_name):
        os.makedirs(root_name)
    return

class Counter:
    def __init__(self):
        self.count, self.sum, self.avg = 0, 0, 0
        return

    def updata(self, value, num_updata=1):
        self.count += num_updata
        self.sum += value * num_updata
        self.avg = self.sum / self.count
        return

    def clear(self):
        self.count, self.sum, self.avg = 0, 0, 0
        return

def read_csv(root_csv):
    df = pd.read_csv(root_csv, sep='\n')
    df = df.values
    df = np.squeeze(df)
    return df

def add_window(image, WL=-600, WW=1600):
    WLL = WL - (WW / 2)
    image = (image - WLL) / WW * 255
    image[image < 0] = 0
    image[image > 255] = 255
    image2 = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
    image2[:, :, :] = image[:, :, :]
    return image2
