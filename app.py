'''
app.py
Project: salientime
Created: 2023-08-23 23:22:59
Author: Bill Chen (bill.chen@live.com)
-----
Last Modified: 2024-04-16 16:12:31
Modified By: Bill Chen (bill.chen@live.com)
'''
import gc
import glob
import json
import os
import pickle
import time
from typing import List

import natsort
import numpy as np
import torch
from apiflask import APIFlask
from flask import jsonify, send_file, session
from lru import LRU

from model.stn import *
from salientime import SalientimeCore
from utils import (codegen, dim_reduction_sklearn, load_model, logger,
                   read_config)
from utils.math import norm01, tolist_rounded
from utils.schemas import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create flask app
app = APIFlask('salientime', title='Salientime', docs_ui='elements')
app.config['JSON_SORT_KEYS'] = False
app.config['SESSION_TYPE'] = 'memcached'
app.config['ROUND_DECIMAL'] = read_config('server.round_decimal')
app.secret_key = 'salientime'

# Initialize Model
net = SalienTimeNet(data_dim=1, code_dim=read_config('net.code_size')).to(device).eval()
load_model(net, os.path.join('checkpoints', read_config('net.checkpoint')))
logger.info('STN loaded.')
cores = LRU(10)

# Initialize metadata & numpy uint8
meta = {
    name: json.loads(open(f'datasets/{name}/{name}.json').read()) for name in available_datasets
}
data_uint8 = {
    name: np.load(f'datasets/{name}/uint8.npy') for name in available_datasets
}
tiffs = {
    name: natsort.natsorted(glob.glob(f'datasets/{name}/tiff/*.tiff')) for name in available_datasets
}
logger.info('Meta data and source data (normalized uint8) loaded.')

data_float32 = {}
for d in available_datasets:
    with open(f'datasets/{d}/float32.pkl', 'rb') as f:
        data_float32[d] = pickle.load(f)
logger.info('Float32 data loaded.')


# Utilities
def query_hash(dataset: str, geo_bound: str, ranges: List[int]):
    get_geo_str = lambda geo: f'{geo[0][0]:.1f},{geo[0][1]:.1f}:{geo[1][0]:.1f},{geo[1][1]:.1f}'
    if ranges is None or len(ranges) == 0:
        range_str='rfull'
    else:
        range_str=f'r{ranges[0]}:{ranges[1]}'
    if geo_bound is None or (geo_str := get_geo_str(geo_bound)) == '0.0,0.0:0.0,0.0':
        geo_str='gfull'
    else:
        geo_str=f'g{geo_str}'
    return f'{dataset}_{geo_str}_{range_str}'

@app.get('/')
@app.output(IndexOut)
def index():
    return jsonify(
        message='Salientime server running. See document at /docs.'
    )

@app.get('/datasetinfo/<string:dataset>')
@app.doc(description='Get the information of a dataset.')
@app.input(DatasetInfoIn, location='path')
@app.output(DatasetInfoOut)
def dataset_info(dataset, *args, **kwargs):
    # return jsonify(meta[dataset])
    return jsonify(json.loads(open(f'datasets/{dataset}/{dataset}.json').read()))

@app.post('/load/<string:dataset>')
@app.doc(description='Load the dataset (Performing the latent code calculation, etc.).')
@app.input(LoadIn, location='path')
@app.input(LoadInJson, location='json')
def dataset_load(dataset, json_data, *args, **kwargs):
    begin = time.time()
    session['dataset'] = dataset
    session['geo_bound'] = json_data.get('geo_bound')
    hash_str = query_hash(dataset, session['geo_bound'] , None)
    if hash_str in cores.keys():
        logger.debug('Dataset already loaded.')
        return jsonify(
            message='ok',
            time_elapsed=time.time() - begin,
        )
    session['hash'] = hash_str
    latent = codegen(net, data_uint8[dataset], 128, cache_hash=hash_str)
    logger.debug('Latent code generated.')
    # with open(f'datasets/{dataset}/float32.pkl', 'rb') as f:
    #     original_data = pickle.load(f)
    # data_float32 = np.load(f'datasets/{dataset}/float16.pkl')
    original_data = data_float32[dataset]
    cores[hash_str] = SalientimeCore(latent, original_data, eval=False)
    logger.debug(f'SalientimeCore loaded in {time.time() - begin}s.')
    gc.collect()
    return jsonify(
        message='ok',
        time_elapsed=time.time() - begin,
    )

@app.post('/findframe')
@app.doc(description='Find salient time steps with a given index range.')
@app.input(FindFrameIn, location='json')
@app.output(FindFrameOut)
def find_frame(json_data):
    begin = time.time()
    k = json_data.get('k')
    alpha = json_data.get('alpha')
    beta = json_data.get('beta')
    agg = json_data.get('agg')
    start = json_data.get('start')
    end = json_data.get('end')
    dataset = json_data.get('dataset')
    geo_bound = json_data.get('geo_bound')
    ranges = []
    if start is not None and end is not None:
        ranges = [start, end]
    core_hash_str = query_hash(dataset, geo_bound, None)
    if core_hash_str not in cores.keys():
        return jsonify(message='Dataset not loaded or server restarted recently. Please call /load/<dataset> first.')
    core: SalientimeCore = cores[core_hash_str]
    # frames = core.find_k_mixed(k, ranges, alpha, beta, gamma=0.3, sigma=1.0, agg=agg)
    frames = core.find_k_mixed_fast(k, ranges, alpha, beta, gamma=0.3, sigma=1.0, agg=agg)
    logger.debug(f'Finished in {time.time() - begin}s.')
    return jsonify(
        frames=frames
    )

@app.post('/trend')
@app.doc(description='Get the trend data for the focus range for timeline visualization.')
@app.input(TrendIn, location='json')
@app.output(TrendOut)
def trend_info(json_data):
    dataset = json_data.get('dataset')
    start = json_data.get('start')
    end = json_data.get('end')
    geo_bound = json_data.get('geo_bound')
    latent_only = json_data.get('latent_only')
    if 'step' in json_data:
        step = json_data.get('step')
    else:
        full_len = end - start + 1
        step = 1 if full_len < 200 else full_len // 200
    if latent_only:
        step = 1 if full_len < 300 else full_len // 300
    core_hash_str = query_hash(dataset, geo_bound, None)
    if core_hash_str not in cores.keys():
        return jsonify(message='Dataset not loaded or server restarted recently. Please call /load/<dataset> first.')
    core: SalientimeCore = cores[core_hash_str]
    r = app.config['ROUND_DECIMAL']
    if start is None or end is None or (start == 0 and end == 0):
        start = 0
        end = core.latent_2d.shape[0] - 1

    if latent_only:
        if (end - start) // step <= 20:
            latent_2d = core.latent_2d[start:end:step, :]
        else:
            # step = int(max(1, step // 2))
            # code_segement = core.latent_2d[start:end:step, :]
            code_segement = core.latent_code[start:end:step, :]
            latent_2d = dim_reduction_sklearn(code_segement, verbose=0)
        return jsonify(
            latent_2d=tolist_rounded(norm01(latent_2d, peraxis=True), r),
            step=step
        )
    else:
        return jsonify(
            latent_2d=tolist_rounded(norm01(core.latent_2d[start:end:step, :], peraxis=True), r),
            pairwise_cossim=tolist_rounded(core.cossim_norm[start:end:step, start:end:step], r),
            trend_avg=tolist_rounded(core.meanv[start:end:step], r),
            trend_max=tolist_rounded(core.maxv[start:end:step], r),
            trend_min=tolist_rounded(core.minv[start:end:step], r),
            step=step,
        )

@app.get('/rawdata/<string:dataset>/<int:frame_index>')
@app.input(RawDataIn, location='path')
# @app.input(RawDataInJson, location='json')
@app.doc(
    description='Get the raw GeoTIFF data for a given frame.',
    responses={
        200: {'description': 'The raw GeoTIFF data TIFF file (Possibly cropped, but todo).'},
    }
)
def raw_data(dataset, frame_index, *args, **kwargs):
    return send_file(f'{tiffs[dataset][frame_index]}', mimetype='image/tiff')

if __name__ == '__main__':
    app.run(debug=False, port=read_config('server.port'), host='0.0.0.0')