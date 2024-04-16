'''
config.py
Project: utils
Created: 2023-08-10 00:19:10
Author: Bill Chen (bill.chen@live.com)
-----
Last Modified: 2023-08-27 10:54:52
Modified By: Bill Chen (bill.chen@live.com)
'''

import yaml
import os
from .logger import logger

def read_config(key: str):
    if not os.path.exists('config.yml'):
        logger.error('Config file does not exist. Returning empty value.')
        return ''

    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    keys = key.split('.')
    for k in keys:
        if not k in config:
            logger.error(f'Key {k} not found in config.yml. Returning empty value.')
            return ''
        config = config[k]

    logger.debug(f'Read config: {key} = {config}')
    return config