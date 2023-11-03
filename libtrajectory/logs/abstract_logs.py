"""
Each directory contains the following three files:
model file
params file
evaluate file
"""

import datetime
import os
import shutil
import time

import pandas as pd

from libtrajectory.utils.time_utils import parse_time, datetime_to_mktime


class AbstractLogs(object):

    def __init__(self):
        pass
