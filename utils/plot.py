'''
inspect.py
Project: utils
Created: 2023-08-10 15:46:11
Author: Bill Chen (bill.chen@live.com)
-----
Last Modified: 2024-04-16 17:01:19
Modified By: Bill Chen (bill.chen@live.com)
'''
from typing import List
from osgeo import gdal
import os
import numpy as np
import natsort
import cv2
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colorbar import Colorbar
from . import logger

class DataInspector():

    def __init__(self, data: np.ndarray):
        """Inspect data information

        Args:
            data (np.ndarray, optional): Full data array. [time_step, H, W]
        """
        self.data = data
        self.total_len = data.shape[0]


    def create_video(self, output: str,
                     fps: int=3,
                     colormap: str='rainbow',
                     vrange: List[int] | str='auto',
                     width_output: int=512,) -> None:
        """Create video from raster files

        Args:
            output (str): Output file, including file name.
            fps (int, optional): Frame per second. Defaults to 1.
            colormap (str, optional): Matplotlib colormap. Defaults to 'rainbow'.
            vrange (List[int] | str, optional): Value range. Defaults to 'auto'.
            width_output (int, optional): The output size of the video. Defaults to 512.
        """
        # if the folder that contains output does not exist, create one
        if not os.path.exists(os.path.dirname(output)):
            os.makedirs(os.path.dirname(output), exist_ok=True)

        cmap = plt.get_cmap(colormap)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v') # Requires ffmpeg installed
        height_output = int(self.data.shape[1] * width_output / self.data.shape[2])
        out = cv2.VideoWriter(output, fourcc, fps, (width_output, height_output))

        logger.info('Encoding video...')
        for i in tqdm(range(self.total_len)):
            img = self.data[i]

            if vrange == 'auto':
                img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
            else:
                img_norm = (img - vrange[0]) / (vrange[1] - vrange[0])

            img_colored = cmap(img_norm)[:, :, :3] * 255
            img_colored = img_colored.astype(np.uint8)

            img_resized = cv2.resize(img_colored, (width_output, height_output))
            out.write(img_resized)
        out.release()
        logger.info(f'Video saved to {output}')

    def plot_preview(self, frames: List[int]=[], save_dir:str=None):
        """
        Plot heatmap with given time steps. If no frames is specified, evenly
        sample 10 time steps to plot.

        Args:
            data (np.ndarray):  The np array with shape [time_steps, H, W]
            frames (List[int], optional): _description_. Defaults to [].
        """
        if len(frames) == 0:
            frames = np.linspace(0, self.data.shape[0] - 1, 10, dtype=int)
        data_to_plot = self.data[frames,:,:]
        fig, axes = plt.subplots(int(len(frames) / 5), 5, figsize=(15, 5))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i > data_to_plot.shape[0]:
                break
            ax.imshow(data_to_plot[i], cmap='rainbow')
            ax.set_title(f'{frames[i]}/{self.data.shape[0]}')
        fig.tight_layout()
        plt.show()


def plot_2d(latents_2d: np.ndarray, frames: List[int]=[], title: str='', save_dir: str=None, show: bool=True):
    """Create a scatter plot of the 2D latent code.

    Args:
        latents_2d (np.ndarray): [timesteps, 2]
        frames (List[int]): The selected frames inside the figure.
        title (str): The title of the figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 0.05]}, figsize=(6, 4))
    colors = cm.rainbow(np.linspace(0, 1, len(latents_2d)))
    ax1.set_title(title)
    # Plot each time step with a scatter with lines connecting them
    for i in range(1, len(latents_2d)):
        ax1.plot(latents_2d[i-1:i+1, 0], latents_2d[i-1:i+1, 1], color='gray', linewidth='0.5')
    for i in range(0, len(latents_2d)):
        ax1.scatter(latents_2d[i, 0], latents_2d[i, 1], color=colors[i-1], s=3)
    ax1.scatter(latents_2d[frames, 0], latents_2d[frames, 1], facecolors='none', edgecolors='r', s=15)
    norm = Normalize(vmin=0, vmax=len(latents_2d))
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    cb = Colorbar(cbar_ax, cmap=cm.rainbow, norm=norm, orientation='vertical')
    cb.set_label('Time steps')
    # Add horizon bar at the bottom
    ax2.bar(frames, [1]*len(frames), color='red')
    if save_dir != None:
        fig.savefig(save_dir, dpi=300, bbox_inches='tight')
    if show:
        fig.show()

def plot_image(image: np.ndarray):
    """Plot a single image

    Args:
        image (np.ndarray): The image array, can be [x, H, W]
    """
    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.show()

def plot_colorize_png(png_path: str, save_dir: str=None, show: bool=True, colormap='turbo'):
    """Plot colorized png file

    Args:
        png_path (str): The path to the png file
        save_dir (str, optional): The path to save the figure. Defaults to None.
        show (bool, optional): Whether to show the figure. Defaults to True.
    """
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    image = io.imread(png_path)
    image = image[:, :, 0]
    image = image / 255
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=colormap)
    ax.axis('off')
    if save_dir != None:
        fig.savefig(save_dir, bbox_inches='tight', pad_inches=0)
    if show:
        fig.show()