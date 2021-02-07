import logging
from os.path import splitext
from os import listdir

import numpy as np
import torch
from torch.utils.data import Dataset

from glob import glob
from PIL import Image



class Data_set(Dataset):
    def __init__(self, dir_fake, dir_real, scale=1):
        self.dir_fake = dir_fake
        self.dir_real = dir_real
        self.scale=scale
        self.name = [splitext(file)[0] for file in listdir(
            dir_fake) if not file.startswith('.')]

    def __len__(self):
        return len(self.name)
    @classmethod
    def resize(cls, img, scale):
        W, H = img.size
        new_W = int(W*scale)
        new_H = int(H*scale)
        new_img = img.resize((new_W, new_H))
        a = np.array(new_img)
        if len(a.shape) == 2:
            a = np.expand_dims(a, axis=2)
        a = a.transpose((2, 0, 1))
        if a.max() > 1:
            a = a/255
        return a

    def __getitem__(self, i):
        x = self.name[i]
        file_fake = glob(self.dir_fake + x + '.*')
        file_real = glob(self.dir_real + x + '_real.*')
        try:
            img = Image.open(file_fake[0])
            mask = Image.open(file_real[0])
        except:
            print(fike_fake)
            print(file_real)
        fake = self.resize(fake, self.scale)
        real = self.resize(real, self.scale)

        return {
            'fake': torch.from_numpy(fake).type(torch.FloatTensor),
            'real': torch.from_numpy(real).type(torch.FloatTensor),
        }  # return a dict
