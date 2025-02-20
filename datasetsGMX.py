 # coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat
import numpy as np
import os

# 如果没有disable gcs可能需要代理向google授权(雾)
# disable gcs
from tensorflow_datasets.core.utils import gcs_utils

gcs_utils._is_gcs_disabled = True

# wight path
# WITHT_PATH="/home/b110/code/GMX/input_data/weight1.mat"
WITHT_PATH="/home/b110/code/GMX/input_data/weight1_1mat_12ch.mat"



# choose your mask
MASK = "Idea1GM2"
# mask pathsA
MASK_PATHS = {
    "Idea1GM2": "datasets/GMX/GMX_multi_data_and_mask/train_square/idea1_Square/Square_mask7_70.mat",
    "Idea1GM3": "datasets/GMX/GMX_multi_data_and_mask/train_square/idea1_Square/Square_mask7_80.mat",
    "Idea2GM1": "datasets/GMX/GMX_multi_data_and_mask/train_square/idea2_Car/Car_mask1_R4.mat",
    "Idea2GM2": "datasets/GMX/GMX_multi_data_and_mask/train_square/idea2_Car/Car_mask2_R4.mat",
    "Idea2GM3": "datasets/GMX/GMX_multi_data_and_mask/train_square/idea2_Car/Car_mask3_R4.mat",
}
MASK_PATH = MASK_PATHS[MASK]


# 数据集path
# DATASET_ROOT = "/home/b110/code/GMX/datasets/GMX/SIAT500data-singlecoil/data1"
DATASET_ROOT = "/home/b110/code/GMX/datasets/GMX/SIAT500data-12coils-main_cropped_1"
# 测试集path
# TEST_DATASET_ROOT = "/home/b110/code/GMX/datasets/GMX/SIAT500data-singlecoil/data2"
TEST_DATASET_ROOT = "/home/b110/code/GMX/datasets/PD_img"


class GetMRI(Dataset):
    def __init__(self, root, augment=None):
        super().__init__()
        self.data_names = np.array([root + "/" + x for x in os.listdir(root)])
        self.augment = None

    def k2wgt(self, X, W):
        # 有些数据会出错，可能会引起报错
        result = np.multiply(X, W)
        return result
    def k2mask(self, X, W):
        result = np.multiply(X, W)
        return result

    def __getitem__(self, index):
        # 图片数据字段可能取不同的值，e.g.'Img','PD_img'
        # siat_input = loadmat(self.data_names[index])['Img']
        # siat_input = loadmat(self.data_names[index])['PD_img']
        siat_input = loadmat(self.data_names[index])['modifiedData']    # siat_input 256*256*12
        # siat = np.array(siat_input[:, :, 0:2], dtype=np.float32)
        siat = np.array(siat_input[:, :], dtype=np.float32)

        siat_complex = siat[:, :, 0] + 1j * siat[:, :, 1]
        siat_kdata = np.fft.fft2(siat_complex)
        siat_kdata = np.fft.fftshift(siat_kdata)

        # weight = loadmat('/home/b110/code/GMX/input_data/weight1.mat')['weight']
        weight = loadmat(WITHT_PATH)['weight']
        kdata_w = self.k2wgt(siat_kdata, weight)    # kdata_w dtype=complex128

        # add mask
        mask = loadmat(MASK_PATH)['mask']   # mask 256*256*1, dtype=int8
        kdata_w_m = self.k2mask(kdata_w, mask)


        siat_temp = np.zeros((256, 256, 6))
        kdata = np.array(siat_temp, dtype=np.float32)
        kdata[:, :, 0] = np.real(kdata_w_m)
        kdata[:, :, 1] = np.imag(kdata_w_m)
        kdata[:, :, 2] = np.real(kdata_w_m)
        kdata[:, :, 3] = np.imag(kdata_w_m)
        kdata[:, :, 4] = np.real(kdata_w_m)
        kdata[:, :, 5] = np.imag(kdata_w_m)

        kdata = kdata.transpose((2, 0, 1))  # （6，256，256）

        return kdata

    def __len__(self):
        return len(self.data_names)

    def __len__(self):
        return len(self.data_names)


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
    image = tf.image.resize(image,
                            size=(resolution, resolution),
                            antialias=True,
                            method=tf.image.ResizeMethod.BICUBIC)
    return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
    # Compute batch size for this worker.
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    print(batch_size)
    if batch_size % jax.device_count() != 0:
        raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                         f'the number of devices ({jax.device_count()})')

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1

    # Create dataset builders for each dataset.
    if config.data.dataset == 'CIFAR10':
        dataset_builder = tfds.builder('cifar10')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(
                img, [config.data.image_size, config.data.image_size],
                antialias=True)

    elif config.data.dataset == 'SVHN':
        dataset_builder = tfds.builder('svhn_cropped')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(
                img, [config.data.image_size, config.data.image_size],
                antialias=True)

    elif config.data.dataset == 'CELEBA':
        #dataset_builder = tfds.builder('celeb_a')
        train_split_name = 'train'
        eval_split_name = 'validation'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = central_crop(img, 140)
            img = resize_small(img, config.data.image_size)
            return img

    #elif config.data.dataset == 'LSUN':
    #  dataset_builder = tfds.builder(f'lsun/{config.data.category}')
    #  train_split_name = 'train'
    #  eval_split_name = 'validation'

    #  if config.data.image_size == 128:
    #    def resize_op(img):
    #      img = tf.image.convert_image_dtype(img, tf.float32)
    #      img = resize_small(img, config.data.image_size)
    #      img = central_crop(img, config.data.image_size)
    #      return img

    #  else:
    #    def resize_op(img):
    #      img = crop_resize(img, config.data.image_size)
    #      img = tf.image.convert_image_dtype(img, tf.float32)
    #      return img

    elif config.data.dataset == 'LSUN':
        #dataset_builder = tfds.builder(f'lsun/{config.data.category}')
        train_split_name = 'train'
        eval_split_name = 'validation'

        def resize_op(img):
            img = crop_resize(img, config.data.image_size)
            img = tf.image.convert_image_dtype(img, tf.float32)
            return img

    elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
        dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
        train_split_name = eval_split_name = 'train'

    else:
        raise NotImplementedError(
            f'Dataset {config.data.dataset} not yet supported.')

    # Customize preprocess functions for each dataset.
    if config.data.dataset in ['FFHQ', 'CelebAHQ']:

        def preprocess_fn(d):
            sample = tf.io.parse_single_example(
                d,
                features={
                    'shape': tf.io.FixedLenFeature([3], tf.int64),
                    'data': tf.io.FixedLenFeature([], tf.string)
                })
            data = tf.io.decode_raw(sample['data'], tf.uint8)
            data = tf.reshape(data, sample['shape'])
            data = tf.transpose(data, (1, 2, 0))
            img = tf.image.convert_image_dtype(data, tf.float32)
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (tf.random.uniform(img.shape, dtype=tf.float32) +
                       img * 255.) / 256.
            return dict(image=img, label=None)

    else:

        def preprocess_fn(d):
            """Basic preprocessing function scales data to [0, 1) and randomly flips."""
            img = resize_op(d['image'])
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (tf.random.uniform(img.shape, dtype=tf.float32) +
                       img * 255.) / 256.

            return dict(image=img, label=d.get('label', None))

    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)
        if isinstance(dataset_builder, tfds.core.DatasetBuilder):
            dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(split=split,
                                            shuffle_files=True,
                                            read_config=read_config)
        else:
            ds = dataset_builder.with_options(dataset_options)
        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size)

    #train_ds = create_dataset(dataset_builder, train_split_name)
    #eval_ds = create_dataset(dataset_builder, eval_split_name)

    dataset = GetMRI(
        root=DATASET_ROOT)
    test_dataset = GetMRI(
        root=TEST_DATASET_ROOT)

    train_ds = DataLoader(dataset,
                          batch_size=config.training.batch_size,
                          shuffle=True,
                          num_workers=4)

    eval_ds = DataLoader(test_dataset,
                         batch_size=config.eval.batch_size,
                         shuffle=True,
                         num_workers=4,
                         drop_last=True)

    return train_ds, eval_ds  #, dataset_builder
