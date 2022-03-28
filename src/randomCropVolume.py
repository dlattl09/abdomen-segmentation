import numpy as np
import torch
import os
import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import nibabel as nib
from nilearn.image import resample_img
from random import shuffle
from torch import nn
import math

class RandomCropVolume(object):
    def __init__(self, size, image_field, label_field=None):
        self.size = size
        self.image_field = image_field
        self.label_field = label_field
    def pad_to_minimal_size(self, image, pad_mode='constant'):
        pad = self.size - np.asarray(image.shape[1:4]) + 1
        pad[pad < 0] = 0

        pad_before = np.floor(pad / 2.).astype(int)
        pad_after = (pad - pad_before).astype(int)

        pad_vector = [(0, 0)]
        for i in range(image.ndim - 1):
            if i < 3:
                pad_vector.append((pad_before[i], pad_after[i]))
            else:
                pad_vector.append((0, 0))
        image = np.pad(array=image, pad_width=pad_vector, mode=pad_mode)
        return image, pad_before, pad_after
    def __call__(self, data):
        image_entries = []
        label_entries = []
        image_field = self.image_field
        label_field = self.label_field

        for image_entry, label_entry in zip(data[image_field], data[label_field]):
            assert np.all(np.asarray(image_entry.shape[1:4]) == np.asarray(label_entry.shape[1:4]))
            #assert np.all(np.asarray(image_entry.shape[1]))

            image_entry, pad_before, pad_after = self.pad_to_minimal_size(image_entry, pad_mode='constant')
            label_entry, _, _ = self.pad_to_minimal_size(label_entry, pad_mode='constant')
            
            img_size = copy.deepcopy(self.size)
            padded_img = copy.deepcopy(image_entry.shape)

            for i in range(2):
                end_pxx=random.randint(img_size[0],padded_img[1])
                start_pxx=(end_pxx-img_size[0])
                
                end_pxy=random.randint(img_size[1],padded_img[2])
                start_pxy=(end_pxy-img_size[1])
                
                end_pxz=random.randint(img_size[2],padded_img[3])
                start_pxz=(end_pxz-img_size[2])
                
                #assert (end_pxx,end_pxy,end_pxz <= padded_img[1],padded_img[2],padded_img[3])
                #assert (start_pxx,start_pxy,start_pxz >= 0)
            
                image_patch = image_entry[:, start_pxx:end_pxx, start_pxy:end_pxy, start_pxz:end_pxz]
                label_patch = label_entry[:, start_pxx:end_pxx, start_pxy:end_pxy, start_pxz:end_pxz]
          
           

                assert np.all(np.asarray(image_patch.shape[1:4]) == self.size)
                assert np.all(np.asarray(label_patch.shape[1:4]) == self.size)

                image_entries.append(image_patch)
                label_entries.append(label_patch)
                
                            
        data[self.image_field] = image_entries
        data[self.label_field] = label_entries

        return data
