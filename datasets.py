"""
datasets.py
Project: salientime
Created: 2023-08-02 13:00:34
Author: Bill Chen (bill.chen@live.com)
-----
Last Modified: 2023-08-10 10:39:23
Modified By: Bill Chen (bill.chen@live.com)
"""

from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms, utils, datasets
import torch
import torchvision
import natsort
import numpy as np
import os
from osgeo import gdal

def Perlin():
    """
    Time (512) x 256 x 256
    """
    perlin = np.load('./data/perlin/perlin256.npy') # 512 x 512 x 512
    perlin_tensor = torch.tensor(perlin, dtype=torch.float32)
    dataset = TensorDataset(perlin_tensor)
    return dataset

class GeoTiffDatasetFolder(Dataset):

    def __init__(self, root_dir: str, band: int=1, transform: torchvision.transforms=None):
        """Load dataset from a folder of GeoTiff raster files, sorted with file name.
        The shape for each item is [H, W].

        Args:
            root_dir (str): Root directory
            band (int, optional): The band of GeoTiff file. Starting from 1. Defaults to 1.
            transform (torchvision.transforms, optional): Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(self.root_dir)
        self.file_list = list(filter(lambda x: x.endswith('.tiff'), self.file_list))
        self.file_list = natsort.natsorted(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        geotiff = gdal.Open(file_path, gdal.GA_ReadOnly)
        array = geotiff.ReadAsArray()
        tensor = torch.from_numpy(array) # [H, W]
        tensor = tensor.unsqueeze(0) # [1, H, W]

        if self.transform:
            tensor = self.transform(tensor)

        return tensor