#@title Autoload all modules

from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE

import sampling as sampling_svd

import datasets
import os.path as osp

# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import SIAT_kdata_ncsnpp_test as configs  # 修改config
  model_num = 'checkpoint.pth'
  ckpts = ['ckpt_filename1', 'Idea1_noweight_20','Idea1_noweight_50']
  ckpt_filenames = {
    'ckpt_filename1': './CM_diffusion_Test_DM_MaskRn_TC_0410/exp_total/exp_tu/checkpoints/checkpoint_14.pth',   # 14(8ch) 33(12ch)
    'Idea1_noweight_20': 'exp/GMX/exp_noweight_3m20/checkpoints/checkpoint_16.pth',
    'Idea1_noweight_50': 'exp/GMX/Idea1_50_noweight/checkpoint_12.pth',   # 14(8ch) 33(12ch)
    'Idea1_noweight_70': 'exp/GMX/exp_noweight_3m70/checkpoints/checkpoint_12.pth',
    'Idea1_noweight_120': 'exp/GMX/Idea1_120_noweight/checkpoint_15.pth',
  }

  config = configs.get_config()  
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales) ###################################  sde
  #sde = VESDE(sigma_min=0.01, sigma_max=10, N=100) ###################################  sde
  sampling_eps = 1e-5


batch_size = 8 #@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)


score_models = []



for k in ckpts:
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(),
                                 decay=config.model.ema_rate)
  optimizer = get_optimizer(config, score_model.parameters()) 
  state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)
  state = restore_checkpoint(ckpt_filenames[k], state, config.device)
  ema.copy_to(score_model.parameters())

  score_models.append(score_model)



#@title PC sampling
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
# predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
predictor = sampling_svd.ReverseDiffusionPredictor
# corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
corrector = sampling_svd.LangevinCorrector
snr = 0.075#0.16 #@param {"type": "number"}
n_steps =  1#@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
sampling_fn = sampling_svd.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)




import utils_
import logging
from collections import OrderedDict
import time
#save_path = './result/拼接_possion6_1.8'
save_path = './result/GMX/3model_3sake_series_mask_20_50_Possion_R15'
T2_root = 'datasets/GMX/compare_exp/T1_brain'
T1_root = 'datasets/GMX/compare_exp/T1_brain'
PD_root = 'datasets/GMX/compare_exp/T1_brain'
if not os.path.exists(save_path):
  os.makedirs(save_path)
#save_path = './result/仅t2_gy_possion6_1.5'
if not os.path.exists(save_path):
  os.makedirs(save_path)
utils_.setup_logger(
    "base",
    save_path,
    "test",
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")



# T2_root = './datasets/T2_img'
# T1_root = './datasets/T1_img'
# PD_root = './datasets/PD_img'
dataset = utils_.get_dataset(T2_root, T1_root, PD_root)
dataloader = utils_.get_dataloader(dataset)
test_results = OrderedDict()
test_results["psnr"] = []
test_results["ssim"] = []
test_results["psnr_y"] = []
test_results["ssim_y"] = []

test_results["psnr_zf"] = []
test_results["ssim_zf"] = []
test_times = []



for i, test_data in enumerate(dataloader):
  if i == 10:
    print(f'前{i}张图测试完成')
    break
  img_path = test_data["T2_path"][0]
  img_name = os.path.splitext(os.path.basename(img_path))[0]
  tic = time.time()
  x, n = sampling_fn(score_models[0], score_models[1], score_models[2], test_data, img_name, save_path)
  toc = time.time()
  test_time = toc - tic
  test_times.append(test_time)
  max_psnr = n["psnr"]
  max_psnr_ssim = n["ssim"]
  psnr_zf = n["zf_psnr"]
  ssim_zf = n["zf_ssim"]
  
  test_results["psnr"].append(max_psnr)
  test_results["ssim"].append(max_psnr_ssim)
  test_results["psnr_zf"].append(psnr_zf)
  test_results["ssim_zf"].append(ssim_zf)


  logger.info(
      "img:{:15s} - PSNR: {:.2f} dB; SSIM: {:.4f}  *****  零填充: PSNR: {:.2f} dB; SSIM: {:.4f} ***** time: {:.4f} s".format(
          img_name, max_psnr, max_psnr_ssim, psnr_zf, ssim_zf, test_time
      )
  )
ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
ave_psnr_zf = sum(test_results["psnr_zf"]) / len(test_results["psnr_zf"])
ave_ssim_zf = sum(test_results["ssim_zf"]) / len(test_results["ssim_zf"])
ave_time = np.mean(test_times)
logger.info(
    "----Average PSNR/SSIM results----\n\tPSNR: {:.2f} dB; SSIM: {:.4f}*****  零填充: PSNR: {:.2f} dB; SSIM: {:.4f} ***** Average_time: {:.4f}s\n".format(
        ave_psnr, ave_ssim, ave_psnr_zf, ave_ssim_zf, ave_time
    )
)


