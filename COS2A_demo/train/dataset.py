import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join, isfile
import scipy.io as sio
import random
import torch
import torch.nn.functional as F

def is_mat_file(filename):
    return filename.endswith(".mat")

def load_data(filepath):
    """Load HSI and MSI data from .mat file"""
    data = sio.loadmat(filepath)
    # Based on the actual data structure where X is HSI and Y is MSI
    hsi = data['X']  # HSI data (256, 256, 172)
    msi = data['Y']  # MSI data (256, 256, 12)
    
    hsi = torch.tensor(hsi).float()
    msi = torch.tensor(msi).float()
    return hsi, msi

class HSIDataset(data.Dataset):
    """Dataset for HSI super-resolution training"""
    def __init__(self, dataset_dir, patch_size=64, data_augmentation=True, input_transform=None):
        super(HSIDataset, self).__init__()
        
        self.patch_size = patch_size
        self.data_augmentation = data_augmentation
        
        # Get all .mat files in the training directory
        self.mat_files = [join(dataset_dir, f) for f in listdir(dataset_dir) if is_mat_file(f)]
        
        self.input_transform = input_transform
        self.num_samples = 20000  # Number of random patches to generate
        
    def __getitem__(self, index):
        # Determine which file to use (cycling through all files)
        file_idx = index % len(self.mat_files)
        filepath = self.mat_files[file_idx]
        
        # Load the data
        hsi, msi = load_data(filepath)  # HSI: (256, 256, 172), MSI: (256, 256, 12)
        
        # Extract random patch
        h, w = hsi.shape[0], hsi.shape[1]
        x = random.randint(0, h - self.patch_size)
        y = random.randint(0, w - self.patch_size)
        
        hsi_patch = hsi[x:x+self.patch_size, y:y+self.patch_size, :]
        msi_patch = msi[x:x+self.patch_size, y:y+self.patch_size, :]
        
        # Data augmentation
        if self.data_augmentation:
            # Random rotation
            rot_times = random.randint(0, 3)
            if rot_times > 0:
                hsi_patch = torch.rot90(hsi_patch, rot_times, [0, 1])
                msi_patch = torch.rot90(msi_patch, rot_times, [0, 1])
            
            # Random flip
            if random.random() > 0.5:
                hsi_patch = torch.flip(hsi_patch, [0])
                msi_patch = torch.flip(msi_patch, [0])
            
            if random.random() > 0.5:
                hsi_patch = torch.flip(hsi_patch, [1])
                msi_patch = torch.flip(msi_patch, [1])
        
        # Convert to channels-first format (C, H, W)
        hsi_patch = hsi_patch.permute(2, 0, 1)
        msi_patch = msi_patch.permute(2, 0, 1)
        
        return msi_patch, hsi_patch
    
    def __len__(self):
        return self.num_samples

class HSITestDataset(data.Dataset):
    """Dataset for HSI super-resolution testing"""
    def __init__(self, dataset_dir, input_transform=None):
        super(HSITestDataset, self).__init__()
        
        # Get all .mat files in the test directory
        self.mat_files = [join(dataset_dir, f) for f in listdir(dataset_dir) if is_mat_file(f)]
        self.filenames = [f for f in listdir(dataset_dir) if is_mat_file(f)]
        
        self.input_transform = input_transform
        
    def __getitem__(self, index):
        filepath = self.mat_files[index]
        filename = self.filenames[index]
        
        # Load the data
        hsi, msi = load_data(filepath)  # HSI: (256, 256, 172), MSI: (256, 256, 12)
        
        # Convert to channels-first format (C, H, W)
        hsi = hsi.permute(2, 0, 1)
        msi = msi.permute(2, 0, 1)
        
        return msi, hsi, filename
    
    def __len__(self):
        return len(self.mat_files)
