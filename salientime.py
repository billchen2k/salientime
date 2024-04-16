'''
salientime.py
Project: salientime
Created: 2023-08-10 17:45:51
Author: Bill Chen (bill.chen@live.com)
-----
Last Modified: 2024-04-16 23:52:14
Modified By: Bill Chen (bill.chen@live.com)
'''

from ast import Tuple
import os
from typing import List

import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import bottleneck as bn
from tqdm import tqdm

import utils
from utils import dim_reduction, dim_reduction_sklearn, logger
from utils.geo import load_geotiff_folder, load_png_folder
from utils.math import norm01
from utils import codegen

from cpp.select_k import select_k

class SalientimeCore():
    def __init__(self, latent_code: np.ndarray, original_data: np.ndarray = None, eval: bool = True, dim_reduce: bool = False):
        self.latent_code = latent_code
        self.total_time_step = original_data.shape[0]
        self.frameshape = (original_data.shape[1], original_data.shape[2])
        self._cost_preparation(original_data)
        if not eval or dim_reduce:
            if self.total_time_step > 500:
                self.latent_2d = dim_reduction(latent_code, verbose=0)
            else:
                self.latent_2d = dim_reduction_sklearn(latent_code, verbose=0)
            logger.debug('Dim reduction done.')
        logger.info(
            f'Salientime core loaded. Total time step: {self.total_time_step}')
        if eval:
            # Persist original data for evaluation. Should not be enabled when in production.
            self.original_data = original_data

    def linear_interp(self, frames: List[int]) -> np.ndarray:
        """Perform linear interpolation to the given data frames and compute the complete data frame.
        Args:
            frames (List[int]): Frames.

        Returns:
            np.ndarray: The reconstructed full data with linear linterpolation.
        """
        reconstruct = np.zeros(
            (frames[-1] - frames[0] + 1, self.frameshape[0], self.frameshape[1]))
        start_frame = frames[0]
        logger.debug(f'Linear interpolating frames {frames}...')
        for i in range(len(frames) - 1):
            for j in range(frames[i + 1] - frames[i]):
                dis = frames[i + 1] - frames[i]
                reconstruct[frames[i] + j - start_frame] = self.original_data[frames[i]] * (1 - j / dis) +\
                    self.original_data[frames[i + 1]] * j / dis
        reconstruct[-1, :, :] = self.original_data[frames[-1]]
        return reconstruct

    def evaluate(self, frames: List[int] = []):
        rec = self.linear_interp(frames)
        origin_eval = self.original_data[frames[0]: frames[-1] + 1, :, :]

        # Remove steps contained in frames
        # origin_eval = np.delete(origin_eval, np.array(frames) - frames[0], axis=0)
        # rec_eval = np.delete(rec, np.array(frames) - frames[0], axis=0)
        origin_eval = norm01(origin_eval)
        rec_eval = norm01(rec)
        ts = origin_eval.shape[0]
        # Normalize to [0, 1]
        origin_eval = (origin_eval - np.min(origin_eval)) / \
            (np.max(origin_eval) - np.min(origin_eval))
        rec_eval = (rec_eval - np.min(rec_eval)) / \
            (np.max(rec_eval) - np.min(rec_eval))
        logger.debug('Computing metrics...')
        rmse = np.sqrt(np.mean((origin_eval - rec_eval) ** 2))
        psnr = 20 * np.log10(1 / rmse)
        ssims = []
        psnrs = []
        infods = []  # Variation of Information Difference
        for t in range(ts):
            ssims.append(structural_similarity(
                origin_eval[t], rec_eval[t], data_range=1.0, multichannel=False))
            # ssims.append(0)
            # psnrs.append(peak_signal_noise_ratio(origin_eval[t], rec_eval[t], data_range=1.0))
        # psnrnp = np.array(psnrs)
        # psnr = np.mean(psnrnp[psnrnp != np.inf])
        return {
            'rmse': rmse,
            'infod': np.mean(infods),
            'psnr': psnr,
            'ssim': np.mean(ssims)
        }, rec

    def _get_ranges(self, ranges: None | List[int]) -> List[int]:
        if len(ranges) > 1:
            if ranges[0] < 0 or ranges[1] > self.total_time_step or ranges[0] >= ranges[1]:
                logger.error('Invalid range.')
                return [0, self.total_time_step]
            return ranges
        else:
            return [0, self.total_time_step - 1]

    def find_k_even(self, k: int, ranges: List[int] = []):
        logger.debug(f'Finding {k} frames evenly within range {ranges}...')
        start, end = self._get_ranges(ranges)
        frames = np.linspace(start, end, k, dtype=int)
        return frames.tolist()

    def find_k_arc_old(self, k: int | str = 0, alpha: float = 0.1, theta: float = np.pi/2, epsilon: float = 0.04, ranges: List[int] = [], ):
        """_summary_

        Args:
            k (int | str, optional): _description_. Defaults to 0.
            alpha (float, optional): Mixing factor. Defaults to 0.5.
            theta (float, optional): Control accumulate angle threshold. Defaults to np.pi/4.
            epsilon (float, optional): Control accumulate distance threshold. Defaults to 0.05.
            ranges (List[int], optional): _description_. Defaults to [].
        """
        logger.debug(
            f'Finding {k} salient time steps using ARC within range {ranges}...')
        start, end = self._get_anges(ranges)
        latent_norm = (self.latent_code - self.latent_code.min()) / \
            (self.latent_code.max() - self.latent_code.min())
        latent_norm = latent_norm.reshape(
            (latent_norm.shape[0], -1))[start:end, :]

        # [timestep - 1, 2]
        diffs = np.diff(latent_norm, axis=0)
        # [timestep - 1, 2], range \in (0, ~dim)
        arclengths = np.linalg.norm(diffs, axis=1)
        #  [timestep - 1, 2], range \in (0, pi)
        angles = np.arccos(np.clip(np.sum(
            diffs[:-1] * diffs[1:], axis=1) / (arclengths[:-1] * arclengths[1:]), -1, 1))
        logger.debug(
            f'Arc length mean (before norm): {arclengths.mean()}, Angles mean (before norm): {angles.mean() / np.pi}pi')
        # Normalized arclengthm range \in [0, 1]
        # arclengths = norm01(arclengths)
        # angles = norm01(angles)

        frames = [start]
        acc_arclen = 0
        acc_angle = 0
        for i in range(1, latent_norm.shape[0] - 1):
            acc_arclen += arclengths[i - 1]
            acc_angle += angles[i - 1]
            if alpha * (acc_angle / theta) + (1 - alpha) * (acc_arclen / epsilon) >= i - frames[-1]:
                frames.append(i)
                acc_arclen = 0
                acc_angle = 0
        frames.append(end - 1)
        logger.debug(f'Selected {len(frames)} frames: {frames}')
        return frames

    def find_k_arc(self, alpha: float = 0.5, theta: float = 0.5 * np.pi, epsilon: float = 0.2, delta: float = 1.0, ranges: List[int] = []):
        """Find salient frames using method proposed by Porter et al.

        Args:
            alpha (float, optional): Mixing factor of arclength vs angle-based selection. Higher value gives
                more weight to angle-based selection. Defaults to 0.5.
            theta (float, optional): Accumulated angle threshold
            epsilon (float, optional): Accumulated arclength threshold (eucledian based)
            delta: Used to control number of representative frames. Default to 1. Larger value
                will reduce the number frames to be selected.
            ranges (List[int], optional): The range (closed) in which to perform selections. Defaults to [].
        """
        start, end = self._get_ranges(ranges)
        logger.debug(f'Finding salient time steps using ARC within range [{start}, {end}]...')

        n_total = end - start + 1
        latent_norm = norm01(self.latent_2d[start: end + 1, :])
        result = [0]
        # Calculate pairwise eucleian distance
        euclidean = euclidean_distances(latent_norm, latent_norm)
        acc_angle = 0
        for i in range(1, n_total - 1):
            va = latent_norm[i] - latent_norm[i - 1]
            vb = latent_norm[i + 1] - latent_norm[i]
            cur_angle = np.arccos(np.clip(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)), -1, 1))
            acc_angle += cur_angle
            cur_distance = euclidean[result[-1], i]
            if alpha * (acc_angle / theta) + (1 - alpha) * (cur_distance / epsilon) >= delta:
                result.append(i)
                acc_angle = 0
        if end not in result:
            result.append(end)
        result = [f + start for f in result]
        logger.debug(f'Selected {len(result)} frames using ARC: {result}')
        return result

    def find_k_latent(self, k: int | str = 0, ranges: List[int] = [], delta=1.1, gamma=0.1):
        logger.debug(
            f'Finding salient time steps along latent path within range {ranges}...')
        start, end = self._get_ranges(ranges)

        code_range = self.latent_code[start:end + 1, :]
        d = cosine_similarity(code_range, code_range)
        n = code_range.shape[0]  # total number of frames
        # dp[i][j] = The minumum cos sim of sequence ending with the i-th frame, starts with 0 and has length of j
        dp = np.full((n, k + 1), np.inf)
        prev = np.zeros((n, k + 1), dtype=int)
        dp[:, 0] = 0
        dp[0, 1] = 0
        dp[1:, 2] = d[1:, 0]  # The first frame is always selected

        for i in range(n):  # ending frame
            for j in range(1, min(i + 1, k) + 1):  # sequence len
                for t in range(j - 1, i):  # look for previous frames
                    # Distance cost + Structural cost + Statitical Cost
                    cost = dp[t, j - 1] + d[t, i] + gamma * \
                        max(0, -np.log(((i - t) / (n * delta))))
                    if cost < dp[i, j]:
                        dp[i, j] = cost
                        prev[i, j] = t

        frames = np.zeros(k, dtype=int)
        cur = n - 1
        for p in range(k, 0, -1):
            frames[p - 1] = cur
            cur = prev[cur, p]
        return (frames + start).tolist()

    def _cost_preparation(self, original_data):
        if self.latent_code is None:
            logger.warn('Latent code is not loaded.')
        t = self.total_time_step
        data_flatten = original_data.reshape(t, -1)
        # Using bottleneck for faster computatin (probably)
        logger.trace('Calculating max...')
        maxv = bn.nanmax(data_flatten, axis=1)
        logger.trace('Calculating max...')
        minv = bn.nanmin(data_flatten, axis=1)
        logger.trace('Calculating mean...')
        meanv = bn.nanmean(data_flatten, axis=1)
        logger.trace('Calculating pairwise cossim...')
        cossim = cosine_similarity(self.latent_code, self.latent_code)
        logger.trace('Done. Normalizing...')
        maxv_norm = norm01(maxv)
        minv_norm = norm01(minv)
        meanv_norm = norm01(meanv)
        cossim_norm = norm01(cossim)
        self.maxv = maxv
        self.minv = minv
        self.meanv = meanv
        self.cossim_norm = cossim_norm

        self.cost_cossim = cossim_norm
        self.cost_maxv = - \
            np.tanh(np.abs(maxv_norm[:, np.newaxis] - maxv_norm)) + 1
        self.cost_minv = - \
            np.tanh(np.abs(minv_norm[:, np.newaxis] - minv_norm)) + 1
        self.cost_meanv = - \
            np.tanh(np.abs(meanv_norm[:, np.newaxis] - meanv_norm)) + 1
        # n = self.total_time_step
        # self.cost_distance = -np.tanh(np.abs(np.arange(self.total_time_step)[:, np.newaxis] - np.arange(self.total_time_step))) + 1
        self.cost_distance = - \
            np.tanh(np.arange(self.total_time_step) / self.total_time_step) + 1
        logger.debug('Cost preparation done.')

    def _cost_latent(self, i, j):
        return self.cost_cossim[i, j]
        return (self.cossim_norm[i, j])

    def _cost_distance(self, i, j, n, sigma: float = 1.0):
        return self.cost_distance[abs(i - j)]
        # return max(0, -np.log((np.abs(i - j) / (n * sigma))))
        # return -np.tanh((np.abs(i - j) / (n * sigma))) + 1

    def _func_cost_statistical(self, agg: str = 'max'):
        # Todo: optimize to avoid multiple string comparison
        if agg == 'max':  # The larger for the variation of max, the smaller the cost
            return lambda i, j: self.cost_maxv[i, j]
            return -(np.abs(self.maxv_norm[i] - self.maxv_norm[j])) + 1
        elif agg == 'min':
            return lambda i, j: self.cost_minv[i, j]
            return -(np.abs(self.minv_norm[i] - self.minv_norm[j])) + 1
        elif agg == 'mean' or agg == 'avg':
            return lambda i, j: self.cost_meanv[i, j]
            return -(np.abs(self.meanv_norm[i] - self.meanv_norm[j])) + 1
        else:
            logger.error(
                f'Invalid aggregation {agg}. Returning statistical cost 0.')
            return 0

    def _matrix_cost_statistical(self, agg: str = 'max'):
        if agg == 'max':  # The larger for the variation of max, the smaller the cost
            return self.cost_maxv
        elif agg == 'min':
            return self.cost_minv
        elif agg == 'mean' or agg == 'avg':
            return self.cost_meanv
        else:
            logger.error(
                f'Invalid aggregation {agg}. Returning statistical cost 0.')
            return 0

    def find_k_mixed(self, k: int, ranges: List[int] = [], alpha=0.5, beta=0.5, gamma=0.1, sigma=1.0, agg='max', nostep=False):
        """Mixed selection of frames

        Args:
            k (int): Number of frames
            ranges (List[int], optional): Range to search. Defaults to [].
            alpha (float, optional): Structural Cost
            beta (float, optional): Statistical Cost
            gamma (float, optional): Distance Cost to 0.1.
            sigma (float, optional): Hyper Parameter for Distance Cost to 1.1.
            agg (str, optional): Aggregation method. One of max, min, mean
        """

        # self.cost_distance = -np.tanh(np.arange(self.total_time_step) / (sigma * self.total_time_step)) + 1
        self.cost_distance = - \
            np.tanh(np.arange(self.total_time_step) /
                    (self.total_time_step / (k - 1))) + 1

        start, end = self._get_ranges(ranges)
        logger.debug(
            f'Finding frames within range {ranges}: alpha={alpha}, beta={beta}, gamma={gamma}, sigma={sigma}...')
        _cost_statistical = self._func_cost_statistical(agg)
        n = end - start + 1  # total number of frames

        if nostep:
            step = 1
        else:
            step = 1 if n <= 1000 else n // 1000

        # dp[i][j] = The minumum cos sim of sequence ending with the i-th frame, starts with 0 and has length of j
        dp = np.full((n, k + 1), np.inf)
        prev = np.zeros((n, k + 1), dtype=int)
        dp[:, 0] = 0
        dp[0, 1] = 0

        def fcost(i, j): return alpha * self._cost_latent(i, j) +\
            beta * _cost_statistical(i, j) +\
            gamma * self._cost_distance(i, j, n, sigma)

        # The first frame is always selected
        for i in range(1, n):
            dp[i, 2] = fcost(start, start + i)
        if step > 1:
            dp[1:step + 1, 2] = fcost(start, start + i + step)  # stepped

        if step == 1:
            for i in range(n):  # ending frame
                for j in range(1, min(i + 1, k) + 1):  # sequence len
                    for t in range(j - 1, i):  # look for previous frames
                        # Structural cost + Statitical Cost + Distance cost
                        # cost = dp[t, j - 1] + fcost(t, i)
                        # Without lambda function for higher performance
                        cost = dp[t, j - 1] + \
                            alpha * self.cost_cossim[start + t, start + i] + \
                            beta * _cost_statistical(start + t, start + i) + \
                            gamma * self.cost_distance[abs(i - t)]
                        if cost < dp[i, j]:
                            dp[i, j] = cost.item()
                            prev[i, j] = t

            frames = np.zeros(k, dtype=int)
            cur = n - 1
            # cur = i # stepped
            for p in range(k, 0, -1):
                frames[p - 1] = cur
                cur = prev[cur, p]
        else:
            for i in range(n // step):
                for j in range(1, min(i + 1, k) + 1):
                    for t in range(j - 1, i):
                        cost = dp[t, j - 1] + \
                            alpha * self.cost_cossim[start + t * step, start + i * step] + \
                            beta * _cost_statistical(start + t * step, start + i * step) + \
                            gamma * self.cost_distance[abs(i - t) * step]
                        if cost < dp[i, j]:
                            dp[i, j] = cost.item()
                            prev[i, j] = t

            frames = np.zeros(k, dtype=int)
            cur = n // step - 1
            # cur = i # stepped
            for p in range(k, 0, -1):
                frames[p - 1] = cur * step
                cur = prev[cur, p]

        logger.debug(
            f'Selected {len(frames)} frames: {(frames + start).tolist()}')
        return (frames + start).tolist()

    def find_k_mixed_fast(self, k: int, ranges: List[int] = [], alpha=0.5, beta=0.5, gamma=0.1, sigma=1.0, agg='max', nostep=False):
        logger.debug(
            f'Finding frames (fast) within range {ranges}: alpha={alpha}, beta={beta}, gamma={gamma}, sigma={sigma}...')
        cost_distance = - \
            np.tanh(np.arange(self.total_time_step) /
                    (self.total_time_step / (k - 1))) + 1
        # cost_distance_mat[i][j] = cost_distance[abs(i - j)]
        i, j = np.indices((self.total_time_step, self.total_time_step))
        cost_distance_mat = cost_distance[np.abs(i - j)]
        cost_latent = self.cost_cossim
        cost_statistical = self._matrix_cost_statistical(agg)
        cost_all = alpha * cost_latent + beta * \
            cost_statistical + gamma * cost_distance_mat
        if nostep:
            maxlen = 1e6
        else:
            maxlen = 500
        start, end = self._get_ranges(ranges)
        cost = cost_all[start:end + 1, start:end + 1]
        frames = select_k(k, cost, int(maxlen))
        frames_final = [f + start for f in frames]
        logger.debug(f'Selected {len(frames)} frames: {frames_final}')
        return frames_final


if __name__ == '__main__':

    # A simple test for evaluating model performance.

    import torch
    from model.stn import SalienTimeNet
    from utils import read_config, load_model

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = './checkpoints/epoch465_recons_loss_ 0.000069'
    net = SalienTimeNet(data_dim=1, code_dim=read_config(
        'net.code_size')).to(device).eval()
    load_model(net, checkpoint)

    # Evaluation config & Load core
    dataset_name = 'global-hs'
    model_name = 'stn512-0908-mix'
    code_size = 512
    original_data = load_png_folder(
        './datasets/global-hs/png', filerange=[0, 240], step=1)
    latents = codegen(net, original_data, cache_hash='globalhs-0_240_1')
    k_range = range(10, 60, 1)
    outputdir = f'./result/salientime/{model_name}.{dataset_name}.k{k_range.start}-{k_range.stop}'
    os.makedirs(outputdir, exist_ok=True)
    core = SalientimeCore(latents, original_data, dim_reduce=True)
    latents_2d = utils.dim_reduction_sklearn(latents)

    k = 13
    alpha = 1.0
    beta = 0.0
    gamma = 0.3
    sigma = 0.8
    agg = 'max'

    frames_arc = core.find_k_arc(alpha=0.2, theta=0.75 * np.pi, epsilon=0.5, delta=2.0)
    karc = len(frames_arc)
    frames_latent = core.find_k_mixed_fast(k=k, alpha=alpha, beta=beta,
                                           gamma=gamma, sigma=sigma, agg=agg)
    frames_even = core.find_k_even(k)
    metric_even, rec_even = core.evaluate(frames_even)
    metric_latent, rec_latant = core.evaluate(frames_latent)
    metric_arc, rec_arc = core.evaluate(frames_arc)
    logger.info(f'LATENT {metric_latent}')
    logger.info(f'EVEN {metric_even}')
    logger.info(f'ARC {metric_arc}')
    logger.info(
        f'LATENT - EVEN {np.array(frames_latent) - np.array(frames_even)}')
    logger.info(
        f'LATENT - ARC  {np.array(frames_latent) - np.array(frames_even)}')