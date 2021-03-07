"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import numpy as np
import os
import cv2
import pickle
import tensorflow as tf
import random
from glob import glob
import librosa

sampling_rate = 16000
num_mcep = 24
frame_period = 5.0
n_frames = 128
lambda_cycle = 10
lambda_identity = 5
ref_level_db=20
min_level_db=-100
hops=192
class Image_data:

    def __init__(self, img_size, channels, dataset_path, domain_list, augment_flag):
        self.img_height = img_size
        self.img_width = img_size
        self.channels = channels
        self.augment_flag = augment_flag

        self.dataset_path = dataset_path
        self.domain_list = domain_list

        self.images = []
        self.shuffle_images = []
        self.domains = []
        self.mdict={}

    def image_processing(self, filename, filename2, domain):
        def my_func(x):
          # x will be a numpy array with the contents of the placeholder below
          #print(x)
          hehe= np.reshape(self.mdict[x.decode("utf-8")],(hops,hops,1))
          return hehe
        #print(filename.astype(str))
        img = tf.compat.v1.py_func(my_func, [filename], tf.float32)
        img2 = tf.compat.v1.py_func(my_func, [filename2], tf.float32)

        # if self.augment_flag :
        #     seed = random.randint(0, 2 ** 31 - 1)
        #     condition = tf.greater_equal(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), 0.5)

        #     augment_height_size = self.img_height + (30 if self.img_height == 256 else int(self.img_height * 0.1))
        #     augment_width_size = self.img_width + (30 if self.img_width == 256 else int(self.img_width * 0.1))

        #     img = tf.cond(pred=condition,
        #                   true_fn=lambda : augmentation(img, augment_height_size, augment_width_size, seed),
        #                   false_fn=lambda : img)

        #     img2 = tf.cond(pred=condition,
        #                   true_fn=lambda: augmentation(img2, augment_height_size, augment_width_size, seed),
        #                   false_fn=lambda: img2)

        return img, img2, domain

    def preprocess(self):
        for domain in self.domain_list:
        #   image_list = glob(os.path.join(self.dataset_path, domain) + '/*.wav')
        #   for i in range(len(image_list)):
        #     y,sr=librosa.load(path=image_list[i], sr=16000)
        #     M=librosa.feature.melspectrogram(y=y, sr=sr, n_mels=hops, n_fft=6*hops,hop_length=hops,win_length=6*hops)
        #     M=M.T
        #     if(M.shape[0]<hops):
        #       N=np.zeros((M.shape[1]-M.shape[0],hops), dtype=np.float32)
        #       M=np.concatenate([M,N],axis=0)
        #     else:
        #       M=M[0:hops]
        #     M=librosa.power_to_db(M.T)-ref_level_db
        #     self.mdict[str(image_list[i])]=normalize(M)
        pickle_off = open ("/content/drive/MyDrive/datafile5.txt", "rb")
        self.mdict = pickle.load(pickle_off)
        # with open('datafile5.txt', 'wb') as fh:
        #   pickle.dump(self.mdict, fh)
        for idx, domain in enumerate(self.domain_list):
            image_list = glob(os.path.join(self.dataset_path, domain) + '/*.wav')
            shuffle_list = random.sample(image_list, len(image_list))
            domain_list = [[idx]] * len(image_list)  # [ [0], [0], ... , [0] ]

            self.images.extend(image_list)
            self.shuffle_images.extend(shuffle_list)
            self.domains.extend(domain_list)

def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1])
    images = tf.cast(images, dtype=out_dtype)
    return images

# def preprocess_fit_train_image(images):
#     images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
#     return images

# def postprocess_images(images):
#     images = adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
#     images = tf.cast(images, dtype=tf.dtypes.uint8)
#     return images

def load_images(image_path, img_size, img_channel):
    # img = np.reshape(np.load(str(image_path))[norm_A],(80,80,1))
    y,sr=librosa.load(path=image_path, sr=16000)
    M=librosa.feature.melspectrogram(y=y, sr=sr, n_mels=hops, n_fft=6*hops,hop_length=hops,win_length=6*hops)
    M=M.T
    if(M.shape[0]<hops):
      N=np.zeros((M.shape[1]-M.shape[0],hops), dtype=np.float32)
      M=np.concatenate([M,N],axis=0)
    else:
      M=M[0:hops]
    M=librosa.power_to_db(M.T)-ref_level_db
    M=normalize(M)
    return np.reshape(M,(hops,hops,1))

def augmentation(image, augment_height, augment_width, seed):
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize(image, [augment_height, augment_width])
    image = tf.image.random_crop(image, ori_image_shape, seed=seed)
    return image

def load_test_image(image_path, img_width, img_height, img_channel):
    y,sr=librosa.load(path=image_path, sr=16000)
    M=librosa.feature.melspectrogram(y=y, sr=sr, n_mels=hops, n_fft=6*hops,hop_length=hops,win_length=6*hops)
    M=M.T
    if(M.shape[0]<hops):
      N=np.zeros((M.shape[1]-M.shape[0],hops), dtype=np.float32)
      M=np.concatenate([M,N],axis=0)
    else:
      M=M[0:hops]
    M=librosa.power_to_db(M.T)-ref_level_db
    M=normalize(M)
    return np.reshape(M,(hops,hops,1))

def normalize(S):
  return np.clip((((S - min_level_db) / -min_level_db)*2.)-1., -1, 1)

def denormalize(S):
  return (((np.clip(S, -1, 1)+1.)/2.) * -min_level_db) + min_level_db

def save_images(images, size, image_path):
    # size = [height, width]
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return ((images+1.) / 2) * 255.0


def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def return_images(images, size) :
    x = merge(images, size)

    return x

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')

def pytorch_xavier_weight_factor(gain=0.02, uniform=False) :

    factor = gain * gain
    mode = 'fan_avg'

    return factor, mode, uniform

def pytorch_kaiming_weight_factor(a=0.0, activation_function='relu') :

    if activation_function == 'relu' :
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu' :
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function =='tanh' :
        gain = 5.0 / 3
    else :
        gain = 1.0

    factor = gain * gain
    mode = 'fan_in'

    return factor, mode

def automatic_gpu_usage() :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def multiple_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),
                 tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
