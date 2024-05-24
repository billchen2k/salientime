from PIL import Image
import numpy as np
import torch
from util import save_image_from_tensor
from skimage import io
from torch.utils.data import Dataset
import tifffile
import glob


class SalienTimeDataset(Dataset):
    def __init__(self, data_dir, data_transform):
        self.data_dir = data_dir
        self.image_name_list = []
        for dir in self.data_dir:
            image_name_list = glob.glob(dir)
            print(len(image_name_list))
            image_name_list = image_name_list[:]
            self.image_name_list += image_name_list
        self.data_transform = data_transform

    
    def __len__(self):
        return len(self.image_name_list)
    
    def __repr__(self):
        return f'data dir: {self.data_dir}\n' \
               f'dataset size: {self.__len__()}'

    def __getitem__(self, idx):
        if self.image_name_list[idx].split('.')[-1] == 'tiff':
            np_image = tifffile.imread(self.image_name_list[idx])
        else:
            assert self.image_name_list[idx].split('.')[-1] == 'png'
            np_image = Image.fromarray(io.imread(self.image_name_list[idx]))
        image_t = self.data_transform(np_image)
        image_t = (image_t - image_t.min()) / (image_t.max() - image_t.min() + 0.000001)
        return {'image': image_t}



if __name__ == '__main__':
    tiff = tifffile.imread('./data/merra2-ts/MERRA2_400_TS_20230101.tiff')
    image_t = torch.from_numpy(tiff).float()
    print(image_t)
    image_t = (image_t - image_t.min()) / (image_t.max() - image_t.min())
    save_image_from_tensor(image_t, './result/test_tiff.png')
