import torch
import tensorflow as tf
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)


def get_psnr_is_declining(num=100):
  '''
    author: yysog
    date: 2023-12-19
    description: Determine whether the indicator(psnr) is declining. This function is added, and other code isn't no changed.
    args:
      num: the number of psnr values to keep track of, default is 30.
  '''
  count = 0
  current_psnr = 0
  def psnr_is_declining(psnr):
    nonlocal count, current_psnr
    if psnr <= current_psnr:
      count += 1
    else:
      count = 0
    current_psnr = psnr
    return False if count < num else True
  return psnr_is_declining
