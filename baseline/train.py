from torch.utils.data import DataLoader
import numpy as np
import logging
import torch
from tqdm import tqdm
from datetime import datetime

from config import opt
import dataset
import feature as feat
from model import UNet3D
import losses
import evalu
import pandas as pd
import os

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class Loss_Saver:
    def __init__(self, moving=False):
        self.loss_list, self.last_loss = [], 0.0
        self.moving = moving
        return

    def updata(self, value):

        if not self.moving:
            self.loss_list += [value]
        elif not self.loss_list:
            self.loss_list += [value]
            self.last_loss = value
        else:
            update_val = self.last_loss * 0.9 + value * 0.1
            self.loss_list += [[update_val]]
            self.last_loss = update_val
        return

    def loss_drawing(self, root_file):
    
        loss_array = np.array(self.loss_list)
        colname = ['loss']
        listPF = pd.DataFrame(columns=colname, data=loss_array)
        listPF.to_csv(f'{root_file}loss.csv', encoding='gbk')

        return


def training():

    Unet = UNet3D(1, 1)
    Unet.load_state_dict(torch.load(opt.root_model_param))

    if opt.use_gpu is True :
        Unet = Unet.cuda()

    loss_fn = losses.DiceLoss()
    optimizer = torch.optim.Adam(Unet.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_LR[0], gamma=opt.decay_LR[1])

    train_dataset = dataset.volume_dataset(range_data=[0, 0.01])
    train_dataLoader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    print('train dataset loading finish!')

    eval_dataset = dataset.volume_dataset(range_data=[0.99, 1])
    eval_dataLoader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False)
    print('eval dataset loading finish!')

    if not os.path.exists(opt.root_exp_file):
        os.makedirs(opt.root_exp_file)
    print('training exp ready!')

    time = datetime.now()
    name_exp = f'{str(time.month).zfill(2)}{str(time.day).zfill(2)}_{str(time.hour).zfill(2)}' \
               f'{str(time.minute).zfill(2)}'

    root_nowexp = f'{opt.root_exp_file}/{name_exp}/'
    feat.create_root(root_nowexp)
    root_param = f'{root_nowexp}param/'
    feat.create_root(root_param)
    logger = get_logger(f'{root_nowexp}exp.log')
    logger.info('start training!')
    losssaver, max_acc = Loss_Saver(), 0.0
    for epoch in range(opt.max_epoch):

        Unet.train()
        epoch_loss = feat.Counter()
        train_dataset.update_num_epoch(epoch)
        for batch_id, (dcm_image, label_image) in tqdm(enumerate(train_dataLoader),
                                                       total=int(len(train_dataset) / opt.batch_size)):
            if dcm_image.shape[0] < opt.batch_size:
                continue

            Input, target = dcm_image.requires_grad_(), label_image
            if opt.use_gpu is True:
                Input, target = Input.cuda(), target.cuda()

            pred = Unet(Input)
            loss = loss_fn(pred, target)

            epoch_loss.updata(float(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losssaver.updata(epoch_loss.avg)

        Unet.eval()

        # dice = evalu.eval_data(Unet, save_pred=False)
        train_dice = evalu.evalu_dice(Unet, train_dataset, train_dataLoader, eval_total=0.2)
        evalu_dice = evalu.evalu_dice(Unet, eval_dataset, eval_dataLoader)

        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t train dice:{:.1f} \t evalu dice:{:.1f}'.format(epoch, opt.max_epoch,
                                                                                                  epoch_loss.avg,
                                                                                                  train_dice,
                                                                                                  evalu_dice))
        scheduler.step()
        
        torch.save(Unet.state_dict(), f'{root_param}EP{epoch}_Dice{evalu_dice}.pkl')

    losssaver.loss_drawing(f'{opt.root_exp_file}/{name_exp}/')
    logger.info('finish training!')
    return
