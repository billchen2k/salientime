'''
eval_salientime.py
Project: salientime
Created: 2023-08-17 19:55:57
Author: Bill Chen (bill.chen@live.com)
-----
Last Modified: 2024-04-16 23:53:00
Modified By: Bill Chen (bill.chen@live.com)
'''
import argparse
import glob
import json
import os
import pickle
import time
from multiprocessing import Pool
from types import SimpleNamespace

from tqdm import tqdm

import matplotlib
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from model.stn import SalienTimeNet
from salientime import SalientimeCore
from utils import codegen, dim_reduction_sklearn, load_model
from utils.config import read_config
from utils.geo import load_geotiff_glob
from utils.logger import logger
from utils.math import norm01

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pickleload(dir: str):
    with open(dir, 'rb') as f:
        return pickle.load(f)

def picklesave(obj, dir: str):
    with open(dir, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

sns.set_style("whitegrid")
matplotlib.rc('font', family='sans-serif')
matplotlib.rcParams.update({
    'font.size': 12,
})

def getsize(w, h):
    scale = 1.3
    return (w * scale, h * scale * 0.85)

class SalientimeExp():

    def _outsub(self, subfolder: str = None):
        if not subfolder:
            os.makedirs(os.path.join(self.out_dir, self.taskcode), exist_ok=True)
            return os.path.join(self.out_dir, self.taskcode)
        else:
            os.makedirs(os.path.join(self.out_dir, self.taskcode, subfolder), exist_ok=True)
            return os.path.join(self.out_dir, self.taskcode, subfolder)

    def __init__(self, args, profile=False):
        self.args = args
        self.profile=profile
        self.profile_stat = {}
        if profile:
            logger.warning('Profile enabled.')
        self.taskcode = f'{args.name}_f{"." .join(map(str, args.frange))}_k{".".join(map(str, args.krange))}_g{args.gamma:.2f}'
        logger.info(f'Loading Salientime Expirement: {self.taskcode}...')
        self.original_data = load_geotiff_glob(args.tiff_glob,
                                               resize=args.resize_ratio,
                                               filerange=args.frange[:2],
                                               step=args.frange[2])
        self.file_list = glob.glob(args.tiff_glob)
        self.file_list = natsort.natsorted(list(map(lambda x: x.split('/')[-1], self.file_list)))[args.frange[0]:args.frange[1]:args.frange[2]]
        logger.info(f'File range: {self.file_list[0]} - {self.file_list[-1]}')
        # self.uint8 = np.array(255 * norm01(self.original_data), dtype=np.uint8)
        self.out_dir = args.out_dir
        checkpoint = args.checkpoint or read_config('net.checkpoint')
        net = SalienTimeNet(data_dim=1, code_dim=read_config('net.code_size')).to(device).eval()
        load_model(net, checkpoint)
        if not profile:
            self.latent_code = codegen(net, self.original_data, batch_size=128, cache_hash=self.taskcode, normalize=True)
        else:
            self.latent_code, latent_duration = codegen(net, self.original_data, batch_size=128, normalize=True, profile=True)
            self.profile_stat['latent'] = latent_duration
        logger.info(f'Run load done.')
        self.profile_stat['preprocess_start'] = time.time()
        self.core = SalientimeCore(self.latent_code, self.original_data, eval=True, dim_reduce=True)
        self.profile_stat['preprocess_end'] = time.time()
        self.latent_2d = dim_reduction_sklearn(self.latent_code)
        self.eval_process = args.eval_process
        self.arck = self.args.arck
        self.arcalpha, self.arctheta, self.arcepsilon, self.arcdelta = self.args.arcparams
        del net # Release net memory

        if profile:
            self.profile_stat['preprocess'] = self.profile_stat['preprocess_end'] - self.profile_stat['preprocess_start']
            self.profile_stat['frames'] = self.original_data.shape[0]
            logger.info(f'Profile: {self.profile_stat}')
            with open(os.path.join(self._outsub(), 'profile.json'), 'w+') as f:
                json.dump(self.profile_stat, f, indent=2)

    def _worker(self, params):
        k = params['k']
        beta = params['beta']
        agg = params['agg']
        metric = params['metric']
        gamma = 0.3
        sigma = 1.0 # not used
        logger.info(f'Running eval worker for k={k}, beta={beta}, agg={agg}...')

        latent_frames=self.core.find_k_mixed_fast(k=k, alpha=1 - beta, beta=beta, gamma=self.args.gamma, sigma=sigma, agg=agg)
        even_frames = self.core.find_k_even(len(latent_frames))
        if metric:
            eval_even_frames, _ = self.core.evaluate(even_frames)
            eval_latent_frames, _ = self.core.evaluate(latent_frames)
        logger.info('EVEN: %s' % eval_even_frames)
        logger.info('LATENT: %s' % eval_latent_frames)
        # replace append with concat
        if metric:
            return({
                'k': k,
                'beta': beta,
                'latent rmse': eval_latent_frames['rmse'],
                'even rmse': eval_even_frames['rmse'],
                'latent psnr': eval_latent_frames['psnr'],
                'even psnr': eval_even_frames['psnr'],
                # 'latent infod': eval_latent_frames['infod'],
                # 'even infod': eval_even_frames['infod'],
                'latent ssim': eval_latent_frames['ssim'],
                'even ssim': eval_even_frames['ssim'],
                'latent frames': latent_frames,
                'even frames': even_frames,
                'diff': np.array(latent_frames) - np.array(even_frames)
            })
        else:
            return({
                'k': k,
                'beta': beta,
                'latent frames': latent_frames,
                'even frames': even_frames,
                'diff': np.array(latent_frames) - np.array(even_frames)
            })

    def run_k(self):
        with Pool(processes=self.eval_process) as pool:
            results = pool.map(self._worker, [{
                'k': k,
                'beta': 0.0,
                'agg': self.args.agg,
                'metric': True
            } for k in range(self.args.krange[0], self.args.krange[1], self.args.krange[2])])
        df_k = pd.DataFrame(results)
        df_k.to_csv(os.path.join(self._outsub(), 'metric_k.csv'))
        picklesave(df_k, os.path.join(self._outsub(), 'metric_k.pkl'))
        self.df_k = df_k

    def run_beta(self):
        with Pool(processes=self.eval_process) as pool:
            results = pool.map(self._worker, [{
                'k': self.args.outk,
                'beta': beta,
                'agg': self.args.agg,
                'metric': True
            } for beta in np.linspace(0, 1, 11)])
        df_beta = pd.DataFrame(results)
        df_beta.to_csv(os.path.join(self._outsub(), 'metric_beta.csv'))
        picklesave(df_beta, os.path.join(self._outsub(), 'metric_beta.pkl'))
        self.df_beta = df_beta

    def run(self):
        if self.args.cache and os.path.exists(os.path.join(self._outsub(), 'metric_k.pkl')):
            logger.info('Cached metric_k.pkl found. Skipping...')
            self.df_k = pickleload(os.path.join(self._outsub(), 'metric_k.pkl'))
        else:
            self.run_k()

        if self.args.cache and os.path.exists(os.path.join(self._outsub(), 'metric_beta.pkl')):
            logger.info('Cached metric_beta.pkl found. Skipping...')
            self.df_beta = pickleload(os.path.join(self._outsub(), 'metric_beta.pkl'))
        else:
            self.run_beta()

        logger.info('Run done.')

    def _arc_worker(self, params):
        arc = self.core.find_k_arc(
            alpha=params['alpha'],
            theta=params['theta'],
            epsilon=params['epsilon'],
            delta=params['delta']
        )
        kprime = len(arc)
        latent = self.core.find_k_mixed_fast(k=kprime, alpha=1.0, beta=0.0, gamma=0.0, sigma=0.0, agg='max', nostep=True)
        even = self.core.find_k_even(kprime)
        arc_metrics, _ = self.core.evaluate(arc)
        latent_metrics, _ = self.core.evaluate(latent)
        even_metrics, _ = self.core.evaluate(even)
        result = {
            'k': kprime,
            'detla': params['delta'],
            'latent rmse': latent_metrics['rmse'],
            'latent ssim': latent_metrics['ssim'],
            'arc rmse': arc_metrics['rmse'],
            'arc ssim': arc_metrics['ssim'],
            'even rmse': even_metrics['rmse'],
            'even ssim': even_metrics['ssim'],
            'latent > arc': 'TRUE' if arc_metrics['rmse'] > latent_metrics['rmse'] else '',
            'arc > even': 'TRUE' if even_metrics['rmse'] > arc_metrics['rmse'] else '',
            'latent > even': 'TRUE' if even_metrics['rmse'] > latent_metrics['rmse'] else ''
        }
        return result

    def run_arc_compare(self):
        logger.info(f'Running comparison with arc: {self.args.arcparams}')
        results = []
        coefficient = np.linspace(0.5, 5, 80)
        with Pool(processes=self.eval_process) as pool:
            results = pool.map(self._arc_worker, [{
                'alpha': self.arcalpha,
                'theta': self.arctheta,
                'epsilon': self.arcepsilon,
                'delta': d
            } for d in [c * self.arcdelta for c in coefficient]])
            # } for d in coefficient])

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self._outsub('arc'), 'arc_compare.csv'))


    def _save_plot(self, fig, outdir: str):
        fig.savefig(outdir, dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

    def plot_selection_k(self):
        logger.info('Plotting selection matrix k...')
        matrix = []
        for index, row in self.df_k.iterrows():
            k_value = row['k']
            lframes = row['latent frames']
            eframes = row['even frames']
            # Create an array of zeros for all frames
            frame_status = np.ones(self.original_data.shape[0]) * 0
            for lf in lframes:
                frame_status[lf] = 1
            for ef in eframes:
                if frame_status[ef] != 1:
                    frame_status[ef] = 0.4
            if k_value == self.args.outk:
                # fill frame_status with -0.1
                frame_status[:] = 0.4
                for lf in lframes:
                    frame_status[lf] = 1

            matrix.append(frame_status)
        fig, ax = plt.subplots(figsize=getsize(4, 2))
        cmap = sns.diverging_palette(255, 0, s=75, l=35, as_cmap=True)
        cmap = sns.diverging_palette(257.2, 18, s=93.9, l=21, as_cmap=True) # https://www.hsluv.org/
        sns.heatmap(matrix, cmap="Blues", cbar=False, linewidth=0, ax=ax)
        ax.set_xticks(np.linspace(0, self.original_data.shape[0] - 1, 3).astype(np.int16))
        ax.set_xticklabels(np.linspace(1, self.original_data.shape[0], 3).astype(np.int16), rotation=0)
        ax.set_yticks(np.arange(0, len(self.df_k), 20))
        ax.set_yticklabels(np.arange(self.df_k['k'].iloc[0], self.df_k['k'].iloc[-1], 20))
        # ax.set_ylabel(r'$k$')
        self._save_plot(fig, os.path.join(self._outsub(), 'selection_k.pdf'))

    def plot_selection_beta(self):
        logger.info('Plotting selection matrix beta...')
        matrix = []
        for index, row in self.df_beta.iloc[::-1].iterrows():
            beta_value = row['beta']
            lframes = row['latent frames']
            # Create an array of zeros for all frames
            frame_status = np.zeros(self.original_data.shape[0])
            for lf in lframes:
                frame_status[lf] = 1
                window = 1 # set to 1 around the window. If in the array border, move left or right accordingly.
                start = max(0, lf - window)
                end = min(self.original_data.shape[0], lf + window + 1)

                if start == 0:
                    end = 2 * window + 1
                if end == self.original_data.shape[0]:
                    start = self.original_data.shape[0] - 2 * window - 1
                # Set the values within the window to 1
                frame_status[start:end] = 1
            matrix.append(frame_status)
        fig = plt.figure(figsize=getsize(3, 1))
        g = sns.heatmap(matrix, cmap='Oranges', cbar=False, linewidth=0)
        g.set_xticks(np.linspace(0, self.original_data.shape[0] - 1, 3).astype(np.int16))
        g.set_xticklabels(np.linspace(1, self.original_data.shape[0], 3).astype(np.int16), rotation=0)
        # g.set_yticks(np.linspace(0, len(self.df_beta), 3).astype(np.int16))
        g.set_yticks([0, 6, 11])
        g.set_yticklabels([1.0, 0.5, 0])
        # g.set_ylabel('beta')
        g.set_ylabel(r'$\beta$')
        # g.set_xlabel(f'Frames')
        # g.yaxis.tick_right()
        self._save_plot(g.get_figure(), os.path.join(self._outsub(), 'selection_beta.pdf'))


    def plot_metrics(self):
        logger.info('Plotting metrics...')
        metrics = ['rmse', 'psnr', 'ssim']
        for m in metrics:
            fig, ax = plt.subplots(figsize=getsize(2, 1))
            sns.lineplot(x=self.df_k['k'],
                         y=self.df_k[f'even {m}'],
                         label='Even', linewidth=0.8, color='tab:cyan', ax=ax)
            sns.lineplot(x=self.df_k['k'],
                         y=self.df_k[f'latent {m}'],
                         label='Struc.', linewidth=1, color='tab:purple', ax=ax)
            ax.legend(fontsize=11, framealpha=0.5)
            ax.set_ylabel(m.upper())
            ax.set_xlabel('')
            # Display grid with a lighter color
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
            # for spine in ax.spines.values():
            #     spine.set_edgecolor('gray')
            self._save_plot(fig, os.path.join(self._outsub(), f'metric_{m}.pdf'))

    def plot_agg_trend(self):
        logger.info('Plotting trends...')
        if self.args.agg == 'max':
            value = self.core.maxv
        if self.args.agg == 'min':
            value = self.core.minv
        if self.args.agg == 'avg':
            value = self.core.meanv

        if 'prmsl' in self.args.name:
            value /= 1000 # kPa
        fig, ax = plt.subplots(figsize=getsize(3, 1))
        sns.lineplot(x=np.arange(len(value)), y=value, ax=ax, linewidth=1, color='darkorange')
        ax.set_ylabel(f'{self.args.unit} ({self.args.agg.upper()})')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        ax.set_xlim(-1, len(value))

        # Only keep 3 ticks
        ax.set_xticks(np.linspace(0, self.original_data.shape[0] - 1, 3).astype(np.int16))
        ax.set_xticklabels(np.linspace(1, self.original_data.shape[0], 3).astype(np.int16), rotation=0)

        frames = self.df_beta[self.df_beta['beta'].apply(lambda x: abs(x - 1.0) < 1e-6)]['latent frames'].iloc[0]
        # Add vertical lines for each value in 'even frames'
        for frame in frames:
            ax.axvline(x=frame, color='maroon', linestyle='-', linewidth=1)
        self._save_plot(fig, os.path.join(self._outsub(), f'aggregation_{self.args.agg}.pdf'))

    def plot_latent_space(self):
        logger.info('Plotting latent space...')
        fig, ax = plt.subplots(figsize=getsize(3, 2))
        # colors = sns.cubehelix_palette(n_colors=len(self.latent_2d), start=.5, rot=-.5, as_cmap=False)
        colors = sns.color_palette('coolwarm', len(self.latent_2d))
        # Plot each time step with a scatter with lines connecting them
        l2d = norm01(self.latent_2d)
        for i in range(1, len(l2d)):
            ax.plot(l2d[i-1:i+1, 0], l2d[i-1:i+1, 1], color='gray', linewidth='0.5', zorder=1)
        for i in range(0, len(self.latent_2d)):
            ax.scatter(l2d[i, 0], l2d[i, 1], color=colors[i-1], s=3, zorder=3, alpha=0.6)
        frames = self.df_k[self.df_k['k'].apply(lambda x: abs(x - self.args.outk) < 1e-6)]['latent frames'].iloc[0]
        ax.scatter(l2d[frames, 0], l2d[frames, 1], facecolors='none', edgecolors='indianred', s=40, linewidth=1.5, zorder=5)
        for i in frames:
            # add frame number
            ax.annotate(str(i + 1),
               xy=(l2d[i, 0], l2d[i, 1]),
               xytext=(0, -5),
               textcoords='offset points',
               ha='center',
               va='top',
               fontsize=11,
               color='black',
               bbox=dict(boxstyle="round,pad=0.1,rounding_size=0.25", edgecolor="none", facecolor="black", alpha=0))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.show()
        # ax.set_xlabel(f'Latent Space (k = {self.args.outk})')
        self._save_plot(fig, os.path.join(self._outsub(), f'latent_space.pdf'))

    def plot_colorized(self):
        logger.info('Plotting colorized images...')
        frames = self.df_k[self.df_k['k'].apply(lambda x: abs(x - self.args.outk) < 1e-6)]['latent frames'].iloc[0]
        cmap_name = self.args.cmap
        if cmap_name[0] == '-':
            cmap_name = cmap_name[1:]
        cmap = sns.color_palette(cmap_name, as_cmap=True)
        self.cmap = cmap
        for idx, f in enumerate(frames):
            if self.args.reverse:
                img = self.original_data[f][::-1]
            else:
                img = self.original_data[f]
            if self.args.cmap[0] == '-':
                img = -img
            aspect_ratio = img.shape[1] / img.shape[0]
            h = 1
            fig, ax = plt.subplots(figsize=getsize(aspect_ratio * h, h))
            sns.heatmap(img, cmap=cmap, robust=True, cbar=False, ax=ax)
            ax.axis('off')
            plt.tight_layout(pad=0)
            out_name = f'{idx}_{f}_{self.file_list[f].replace(".tiff", "")}.png'
            fig.savefig(os.path.join(self._outsub('colorizedk'), out_name), dpi=320, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    def plot_bar(self):
        fig, ax = plt.subplots(figsize=getsize(0.1, 1))
        cmap = self.cmap
        minv = np.min(self.core.minv)
        maxv = np.max(self.core.maxv)
        if 'prmsl' in self.args.name:
            minv /= 1000
            maxv /= 1000
        norm = Normalize(vmin=minv, vmax=maxv)

        cb = Colorbar(ax=ax, mappable=cm.ScalarMappable(norm=norm, cmap='Spectral_r'), orientation='vertical')
        cb.ax.tick_params(labelsize=10)
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        self._save_plot(fig, os.path.join(self._outsub(), f'colorbar.pdf'))

    def plot_matrix(self):
        logger.info('Plotting matrix...')
        limit = 100
        mats = {
            'alpha': self.core.cost_cossim[:limit, :limit],
            'beta': self.core.cost_meanv[:limit, :limit],
        }
        mats['gamma'] = np.zeros_like(mats['alpha'])

        for i in range(limit):
            for j in range(limit):
                mats['gamma'][i, j] = self.core.cost_distance[abs(i - j)]

        mats['mix'] = 0.5 * mats['alpha'] + 0.5 * mats['beta'] + self.args.gamma * mats['gamma']

        cmaps = {
            'alpha': 'Blues',
            'beta': 'Blues',
            'gamma': 'Blues',
            'mix': 'Blues',
        }
        for key, mat in mats.items():
            fig, ax = plt.subplots(figsize=getsize(1.8, 1.5))
            sns.heatmap(mat,
                        cmap=cmaps[key],
                        cbar=True, linewidth=0, ax=ax, vmin=0.2)
            ax.set_xticks(np.linspace(0, limit - 1, 3).astype(np.int16))
            ax.set_xticklabels(np.linspace(1, limit, 3).astype(np.int16), rotation=0)
            ax.set_yticks(np.linspace(0, limit - 1, 3).astype(np.int16)[:-1])
            ax.set_yticklabels(np.linspace(1, limit, 3).astype(np.int16)[:-1], rotation=0)
            self._save_plot(fig, os.path.join(self._outsub('matrix'), f'matrix_{key}.pdf'))

    def plot(self):
        self.plot_selection_k()
        self.plot_selection_beta()
        self.plot_metrics()
        self.plot_agg_trend()
        self.plot_latent_space()
        self.plot_colorized()
        self.plot_bar()
        self.plot_matrix()
        logger.info(f'Plot done. Task code = {self.taskcode}')

    def _worker_profile(self, params):
        k = params['k']
        t = params['t']
        logger.info(f'Running profile worker for k={k}, t={t}...')
        start = time.time()
        result = self.core.find_k_mixed_fast(k=k, ranges=[0, t], alpha=0.8, beta=0.2, gamma=0.3, sigma=0.1, agg='max', nostep=True)
        end = time.time()
        logger.info(f'Done for k={k}, t={t}, time={end - start}')
        return {
            'k': k,
            't': t,
            'time': end - start
        }

    def run_profile(self, k_range = [5, 180], kt = 240, t_range = [60, 1000, 20], tk = 12):
        logger.info('Running profile...')
        profile_k = []
        profile_t = []

        # with Pool(processes=self.eval_process) as pool:
        #     profile_k = pool.map(self._worker_profile, [{
        #         'k': k,
        #         't': kt
        #     } for k in range(k_range[0], k_range[1], 1)])
        for k in tqdm(range(k_range[0], k_range[1], 1)):
            profile_k.append(self._worker_profile({
                'k': k,
                't': kt
            }))

        df_profile_k = pd.DataFrame(profile_k)
        df_profile_k.to_csv(os.path.join(self._outsub(), 'profile_k.csv'))

        # # Do the same for t
        # with Pool(processes=self.eval_process) as pool:
        #     profile_t = pool.map(self._worker_profile, [{
        #         'k': tk,
        #         't': t
        #     } for t in range(t_range[0], t_range[1], 1)])
        for t in tqdm(range(t_range[0], t_range[1], 1)):
            profile_t.append(self._worker_profile({
                'k': tk,
                't': t
            }))
        df_profile_t = pd.DataFrame(profile_t)
        df_profile_t.to_csv(os.path.join(self._outsub(), 'profile_t.csv'))
        logger.info('Profile done.')

class Config(SimpleNamespace):
    def __getitem__(self, key):
        # return None if not exists
        if not hasattr(self, key):
            return None
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


if __name__ == '__main__':

    config_global_hs = Config(
        name='global-hs',
        tiff_glob='./datasets/global-hs/tiff/*.tiff',
        resize_ratio=1.0,
        checkpoint='./checkpoints/ckpt_recloss_0.000069',
        frange=[0, 240, 1],
        krange=[5, 60, 1],
        cmap='-Spectral',
        outk=12,
        arcparams=[0.25, 0.75 * np.pi, 0.4, 1],
        agg='max',
        gamma=0.3,
        cache=True,
        out_dir='./result/exp',
        eval_process=64,
        unit='M',
        reverse=True,
    )

    config_rdi = Config(
        name='rdi',
        tiff_glob='./datasets/rdi/tiff/*.tiff',
        resize_ratio=0.4,
        checkpoint='./checkpoints/ckpt_recloss_0.000069',
        frange=[0, 34, 1],
        krange=[2, 32, 1],
        cmap='-Spectral',
        outk=30,
        agg='min',
        gamma=0.1,
        cache=True,
        out_dir='./result/exp',
        eval_process=36,
        unit='Index',
        reverse=True,
    )

    config = config_global_hs

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=config['name'], help='Name of the dataset')
    parser.add_argument('--tiff_glob', type=str, default=config['tiff_glob'], help='glob pattern specifying the geotiff folder.')
    parser.add_argument('--resize_ratio', type=float, default=config['resize_ratio'], help='Resize ratio [0, 1] to make computation faster.')
    parser.add_argument('--checkpoint', type=str, default=config['checkpoint'], help='Checkpoint name.')
    parser.add_argument('--frange', type=int, nargs=3, default=config['frange'], help='Range of the frame index. [start, end, step]')
    parser.add_argument('--krange', type=int, nargs=3, default=config['krange'], help='Range of the k. [start, end, step]')
    parser.add_argument('--cmap', type=str, default=config['cmap'], help='Colormap name to use.')
    parser.add_argument('--outk', type=int, default=config['outk'], help='The k used for generating colorized raster fiiles.')
    parser.add_argument('--arck', type=int, default=13, help='The k for arc-based selection (baseline method).')
    parser.add_argument('--arcparams', type=list,
                        default=[0.25, 0.75 * np.pi, 0.4, 1], help='Parameters for arc-based selection. [alpha, theta, epsilon, delta]')
    parser.add_argument('--agg', type=str, default=config['agg'], help='Aggregation method for statistical variation. max, min or avg.')
    parser.add_argument('--gamma', type=float, default=config['gamma'], help='Controling the weight of structural cost and statistical cost.')
    parser.add_argument('--unit', type=str, default=config['unit'], help='Unit of the data, will be displayed in the generated plots.')
    parser.add_argument('--cache', type=bool, default=config['cache'], help='Whether to use the cache folder.')
    parser.add_argument('--out_dir', type=str, default=config['out_dir'], help='Output working directory.')
    parser.add_argument('--reverse', type=bool, default=config['reverse'], help='If reverse the colormap.')
    parser.add_argument('--eval_process', type=int, default=config['eval_process'], help='Number of processes for evaluation.')

    c = parser.parse_args()

    configs = [
        config_global_hs
    ]

    # Set general configs for all configs
    for c in configs:
        c.cache = True
        c.out_dir='./result'
        c.eval_process = 18 # reduce eval process
        if not hasattr(c, 'arck'):
            c.arck = 13
        if not hasattr(c, 'arcparams'):
            c.arcparams = [0.4, 0.6 * np.pi, 0.5, 1] # Default value for selection

    #### Running exp ###
    for c in configs:
        exp = SalientimeExp(c)
        exp.run()
        exp.plot()
        exp.plot_latent_space()
        exp.run_arc_compare()
