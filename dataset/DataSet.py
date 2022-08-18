import os
import torch
import glob
import h5py
import random
import numpy as np
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

class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()

        self.name  = dirname.split("/")[-1]
        self.scale = scale
        if "DIV" in self.name:
            self.hr = glob.glob(os.path.join("{}_valid_HR".format(dirname), "*.png"))
            self.lr = glob.glob(os.path.join("{}_valid_LR_bicubic".format(dirname), 
                                             "X{}/*.png".format(scale)))
        else:
            '''
            all_files = glob.glob(os.path.join(dirname, "x{}/*.png".format(scale)))
            self.hr = [name for name in all_files if "HR" in name]
            self.lr = [name for name in all_files if "LR" in name]
            '''
            self.hr = glob.glob(os.path.join(dirname, "hr_test","*.dat"))
            self.lr = glob.glob(os.path.join(dirname, "lr_test","*.dat"))
            
            

        self.hr.sort()
        self.lr.sort()
        #print(len(self.hr))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        if "DIV" in self.name:
            hr = Image.open(self.hr[index])
            lr = Image.open(self.lr[index])
        else:
            hr=np.fromfile(self.hr[index],dtype=np.float32).reshape((1800,3600,1))
            lr=np.fromfile(self.lr[index],dtype=np.float32).reshape((900,1800,1))
        
        filename = self.hr[index].split("/")[-1]

        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)
