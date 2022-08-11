import os
import torch
import h5py
import random
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms

class DataSet(data.Dataset):
    def __init__(self, h5_file_root, patch_size=48, scale=4):
        super(DataSet, self).__init__()
        #self.h5_file = h5_file_root
        self.patch_size = patch_size
        self.scale = scale
        h5f = h5py.File(h5_file_root, "r")


        self.hr = [v[:] for v in h5f["HR"].values()]
        
        self.lr = [v[:] for v in h5f["X2"].values()] 

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
      
        hr = self.hr[index]
        print()

        lr = self.lr[index]
        lr = transforms.ToTensor()(lr)
        hr = transforms.ToTensor()(hr)
        print(lr.size())
        lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
        lr, hr = self.random_vertical_flip(lr, hr)
        lr, hr = self.random_horizontal_flip(lr, hr)
        lr, hr = self.random_rotation(lr, hr)
        return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['HR'])


class ValidDataset(data.Dataset):
    def __init__(self, h5_file_root):
        super(ValidDataset, self).__init__()
        self.h5_root = h5_file_root

    def __getitem__(self, index):
        with h5py.File(self.h5_root, 'r') as f:
            hr = torch.from_numpy(f['X2'][str(index)][::])
            lr = torch.from_numpy(f['X2'][str(index)][::])

            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_root, 'r') as f:
            return len(f['HR'])

