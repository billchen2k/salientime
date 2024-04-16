import numpy as np
import torch
import torchvision
from torchvision import transforms
from sklearn.manifold import TSNE as SKLEARNTSNE
from tsnecuda import TSNE
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from scipy.ndimage import zoom
import os
from .math import norm01
import time
from PIL import Image

from model.stn import SalienTimeNet

from .config import read_config
from .logger import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dim_reduction(data: np.ndarray, verbose: int=1):
    """Reduce the dimension of the given array to 2 using t-SNE.
    """
    try:
        tsne = TSNE(n_components=2, perplexity=50, n_iter=1000, random_seed=42, verbose=verbose) # Much faster than sklearn
        data_2d = tsne.fit_transform(data)
    except:
        logger.warning('CUDA t-SNE failed, using sklearn instead...')
        tsne = SKLEARNTSNE(n_components=2, perplexity=50, n_iter=1000, random_state=42, verbose=verbose)
        data_2d = tsne.fit_transform(data)
    return data_2d

def dim_reduction_sklearn(data: np.ndarray, verbose: int=1):
    """Reduce the dimension of the given array to 2 using t-SNE.
    """
    tsne = SKLEARNTSNE(n_components=2, perplexity=min(data.shape[0], 30), n_iter=1000, random_state=42, verbose=verbose)
    data_2d = tsne.fit_transform(data)
    return data_2d

def save_image_from_tensor(t: torch.tensor, filename: str):
    while len(t.shape) < 4:
        t = t.unsqueeze(0)
    if t.shape[-1] == 1 or t.shape[-1] == 3:
        t = t.permute(0, 3, 1, 2)
    t = t.clone().detach()
    t = t.to(torch.device('cpu'))
    torchvision.utils.save_image(t, filename)

def load_model(net, model_dir):
    net.load_state_dict(torch.load(model_dir)['net'])

def codegen(net: SalienTimeNet, data: np.ndarray, batch_size: int=256, cache_hash: str=None, normalize: bool=False, profile=False):

    class ResizedDataset(Dataset):

        def __init__(self, data: np.ndarray, transform):
            # before: data: [t, h, w]
            self.data = torch.tensor(data).unsqueeze(1)
            # after: data: [t, 1, h, w]
            self.transform = transform

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            frame = self.data[idx]
            frame = self.transform(frame)
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            return {
                'image': frame
            }

    if cache_hash is not None:
        if os.path.exists(f'cache/latents/{cache_hash}.npy'):
            logger.info(f'Returning cached code: {cache_hash}.npy...')
            return np.load(f'cache/latents/{cache_hash}.npy')

    if normalize:
        data = np.array(norm01(data) * 255, dtype=np.uint8)

    dataset = ResizedDataset(data.astype(np.float32), transform=transforms.Compose([
        transforms.Resize([256, 256], antialias=True)
    ]))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = np.zeros((len(dataset), read_config('net.code_size')))
    logger.info('No cache, generating code...')
    t_start = time.time()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            image = batch['image'].to(device)
            codes = net.forward_codeonly(image)
            latents[idx * batch_size:idx * batch_size + batch_size] = codes.squeeze().cpu().numpy()
    t_end = time.time()
    os.makedirs(os.path.join('cache', 'latents'), exist_ok=True)
    np.save(f'cache/latents/{cache_hash}.npy', latents)

    if profile:
        return latents, t_end - t_start
    else:
        return latents