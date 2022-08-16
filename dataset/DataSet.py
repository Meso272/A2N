import os
import torch
import h5py
import random
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms

class DataSet(data.Dataset):
    def __init__(self, h5_file_root, patch_size=64, scale=2,fix_length=0,aug=1):
        super(DataSet, self).__init__()
        #self.h5_file = h5_file_root
        self.patch_size = patch_size
        self.scale = scale
        self.aug=aug
        self.length=fix_length
        h5f = h5py.File(h5_file_root, "r")


        self.hr = [v[:] for v in h5f["HR"].values()]
        
        self.lr = [v[:] for v in h5f["X2"].values()]

        self.hrlen=len(self.hr) 

    @staticmethod
    def random_crop(lr, hr, size, upscale):
        lr_x1 = random.randint(0, lr.shape[2]-size)
        lr_x2 = lr_x1+size
        lr_y1 = random.randint(0, lr.shape[1]-size)
        lr_y2 = lr_y1+size

        hr_x1 = lr_x1*upscale
        hr_x2 = lr_x2*upscale
        hr_y1 = lr_y1*upscale
        hr_y2 = lr_y2*upscale

        lr = lr[:, lr_y1:lr_y2, lr_x1:lr_x2]
        hr = hr[:, hr_y1:hr_y2, hr_x1:hr_x2]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
        return lr, hr

    @staticmethod
    def random_rotation(lr, hr):
        if random.random() < 0.5:
            # (1,2)逆时针，(2, 1)顺时针
            lr = torch.rot90(lr, dims=(2, 1))
            hr = torch.rot90(hr, dims=(2, 1))
        return lr, hr

    def __getitem__(self, index):
        if self.length:
            index= index % self.hrlen
        hr = self.hr[index]
      
        lr = self.lr[index]
        lr = transforms.ToTensor()(lr)
        hr = transforms.ToTensor()(hr)
        
        lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
        if self.aug:
            lr, hr = self.random_vertical_flip(lr, hr)
            lr, hr = self.random_horizontal_flip(lr, hr)
            lr, hr = self.random_rotation(lr, hr)
        return lr, hr

    def __len__(self):
        if self.length:
            return self.length
        else:
            return self.hrlen


class ValidDataset(data.Dataset):
    def __init__(self, h5_file_root):
        super(ValidDataset, self).__init__()
        self.h5_root = h5_file_root

    def __getitem__(self, index):
        with h5py.File(self.h5_root, 'r') as f:
            hr = torch.from_numpy(f['HR'][str(index)][::])
            lr = torch.from_numpy(f['X2'][str(index)][::])

            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_root, 'r') as f:
            return len(f['HR'])

