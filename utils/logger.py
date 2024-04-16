'''
logger.py
Project: utils
Created: 2023-08-10 00:09:33
Author: Bill Chen (bill.chen@live.com)
-----
Last Modified: 2023-08-10 00:33:05
Modified By: Bill Chen (bill.chen@live.com)
'''

import os
import loguru
import sys

logger = loguru.logger
logger.add('logs/log_{time:YYYYMMDD}.log', rotation='1 day')