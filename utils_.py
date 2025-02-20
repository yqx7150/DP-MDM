import os
import random
import sys

import cv2

import numpy as np
import torch
import torch.utils.data as data
import logging
from datetime import datetime
from scipy.io import loadmat



class GetMRI(data.Dataset):
    def __init__(self,root,root2 = None,root3 = None,augment=None):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])
        self.augment = None
        if root2 is not None:
            self.data_names2 = np.array([root2+"/"+x  for x in os.listdir(root2)])
        if root3 is not None:
            self.data_names3 = np.array([root3+"/"+x  for x in os.listdir(root3)])

    def k2wgt(self,X,W):
        result = np.multiply(X,W) 
        return result

    def __getitem__(self,index):
        img_path = self.data_names[index]
        T2_ori_data_img = loadmat(img_path)['data']
        img_path2 = self.data_names2[index]
        T1_ori_data_img = loadmat(img_path2)['data']
        img_path3 = self.data_names3[index]
        PD_ori_data_img = loadmat(img_path3)['data']
        # img_path = self.data_names[index]
        # T2_ori_data_img = loadmat(img_path)['DATA']
        # img_path2 = self.data_names2[index]
        # T1_ori_data_img = loadmat(img_path2)['T1_img']
        # img_path3 = self.data_names3[index]
        # PD_ori_data_img = loadmat(img_path3)['PD_img']
        



        # siat_input=loadmat(self.data_names[index])['T2_label']
        # siat=np.array(siat_input[:,:,0:2],dtype=np.float32)
        #siat_complex = siat[:,:,0]+1j*siat[:,:,1]
        #siat_kdata = np.fft.fft2(siat_complex)
        #siat_kdata = np.fft.fftshift(siat_kdata)
        
        
        # siat_kdata = np.fft.fft2(siat_input)
        # siat_kdata = np.fft.fftshift(siat_kdata)

        # weight=loadmat('./weight_0.379_train.mat')['weight'] 
        # kdata_w = np.zeros((256, 256, 4), dtype=np.complex64)
        
        # kdata_w = self.k2wgt(img_GT_k, weight) 
        
        # k_w = np.stack((np.real(kdata_w), np.imag(kdata_w)), 0)
        
        # k_w = np.repeat(k_w, 3, 0)
        # kdata=kdata.transpose((2,0,1))

        return {'T2_path':img_path, 'T1_path':img_path2, 'T2_img':T2_ori_data_img, 'T1_img':T1_ori_data_img, 'PD_img':PD_ori_data_img}
    
    def __len__(self):
        return len(self.data_names)


def get_dataset(root,root2, root3):

    dataset = GetMRI(root,root2,root3)
    return dataset

def get_dataloader(dataset):

    dataloader = data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
        )
    return dataloader

def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")