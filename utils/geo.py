'''
geo.py
Project: utils
Created: 2023-08-10 16:49:53
Author: Bill Chen (bill.chen@live.com)
-----
Last Modified: 2023-11-16 10:45:00
Modified By: Bill Chen (bill.chen@live.com)
'''
import glob
from osgeo import gdal
import os
import natsort
from .logger import logger
import numpy as np
from skimage import transform
from typing import List
from PIL import Image
import xarray as xr
from multiprocessing import Pool


def geotiff_to_numpy(file: str, band: int=1):
    """Convert geotiff data to numpy array

    Args:
        file (str): Directory to the file
        band (int, optional): The band of the GeoTIFF file, starts with 1.. Defaults to 1.
    """
    # Updates: Using xarray
    ds = xr.load_dataset(file)
    return ds.band_data[band - 1].values

def load_geotiff_folder(root_dir: str, band: int=1, resize='auto', filerange=[800, 1280], step=2) -> np.ndarray:
    """Load geotiff files from a folder and returns an np array with shape [time_step, H, W].

    Args:
        file_dir (str): Directory to the folder
        band (int, optional): The band of the GeoTIFF file, starts with 1.. Defaults to 1.
    """
    # file_list = os.listdir(root_dir)
    # file_list = list(filter(lambda x: x.endswith('.tiff'), file_list))
    file_list = glob.glob(os.path.join(root_dir, '*.tiff'))
    file_list = natsort.natsorted(file_list)
    if filerange:
        file_list = file_list[filerange[0]:filerange[1]:step]
    total_len = len(file_list)
    logger.info(f'Loading {len(file_list)} GeoTIFF files from {root_dir}...')
    data_full = None
    size = None

    data_frames = []

    with Pool(processes=32) as pool:
        logger.debug(f'Lodding data frames in parallel...')
        data_frames = pool.map(geotiff_to_numpy, file_list)

    logger.debug(f'Data frame loaded. Initializing numpy array...')
    for idx, f in enumerate(file_list):
        file_path = os.path.join(root_dir, f)
        data_frame = data_frames[idx]
        if idx == 0:
            if resize == 'auto':
                size = (data_frame.shape[0], data_frame.shape[1])
                data_full = np.zeros((total_len, data_frame.shape[0], data_frame.shape[1]))
            else:
                size = (int(data_frame.shape[0] * resize), int(data_frame.shape[1] * resize))
                data_full = np.zeros((total_len, size[0], size[1]))
        # Resize
        frame_resized = transform.resize(data_frame, size, anti_aliasing=False)
        data_full[idx,:,:] = frame_resized
    logger.info(f'Loaded. Item dimension = {data_full.shape[1]}, {data_full.shape[2]}')
    return data_full


def load_geotiff_glob(pattern: str, band: int=1, resize='auto', filerange=[800, 1280], step=2) -> np.ndarray:
    """Load geotiff files from a folder and returns an np array with shape [time_step, H, W].

    Args:
        file_dir (str): Directory to the folder
        band (int, optional): The band of the GeoTIFF file, starts with 1.. Defaults to 1.
    """
    file_list = glob.glob(pattern)
    file_list = natsort.natsorted(file_list)
    if filerange:
        file_list = file_list[filerange[0]:filerange[1]:step]
    total_len = len(file_list)
    logger.info(f'Loading {len(file_list)} GeoTIFF files from {pattern}...')
    data_full = None
    size = None

    data_frames = []

    with Pool(processes=16) as pool:
        logger.debug(f'Lodding data frames in parallel...')
        data_frames = pool.map(geotiff_to_numpy, file_list)

    logger.debug(f'Data frame loaded. Initializing numpy array...')
    for idx, f in enumerate(file_list):
        data_frame = data_frames[idx]
        if idx == 0:
            if resize == 'auto':
                size = (data_frame.shape[0], data_frame.shape[1])
                data_full = np.zeros((total_len, data_frame.shape[0], data_frame.shape[1]))
            else:
                size = (int(data_frame.shape[0] * resize), int(data_frame.shape[1] * resize))
                data_full = np.zeros((total_len, size[0], size[1]))
        # Resize
        frame_resized = transform.resize(data_frame, size, anti_aliasing=False)
        data_full[idx,:,:] = frame_resized
    logger.info(f'Loaded. Item dimension = {data_full.shape[1]}, {data_full.shape[2]}')
    return data_full

def load_png_folder(root_dir: str, resize: 'str' | List['str']='auto', filerange=None, step:int = 1) -> np.ndarray:
    file_list = glob.glob(os.path.join(root_dir, '*.png'))
    file_list = natsort.natsorted(file_list)
    if filerange:
        file_list = file_list[filerange[0]:filerange[1]:step]
    data_full = None
    size = None
    if len(file_list) == 0:
        logger.warn(f'No PNG files found in {root_dir}')
    for idx, f in enumerate(file_list):
        img = Image.open(f)
        img = np.array(img)
        if idx == 0:
            if resize == 'auto':
                size = (img.shape[0], img.shape[1])
                data_full = np.zeros((len(file_list), img.shape[0], img.shape[1]))
            else:
                size = resize
                data_full = np.zeros((len(file_list), size[0], size[1]))
        # Resize
        frame_resized = transform.resize(img, size, anti_aliasing=False)
        data_full[idx,:,:] = frame_resized
    logger.info(f'Loaded. Item dimension = {img.shape[0]}, {img.shape[1]}')
    return data_full