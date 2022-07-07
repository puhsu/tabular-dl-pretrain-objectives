# flake8: noqa
# type: ignore
import gc
import itertools
import json
import os
import shutil
import statistics
import sys
import typing as ty
from collections import Counter
from copy import deepcopy
from dataclasses import *
from pathlib import Path
from pprint import pprint
from types import LambdaType, ModuleType

import holoviews as hv
import numpy as np
import pandas as pd
import pyarrow.csv
import rtdl
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero
from icecream import ic
from torch import Tensor
from tqdm.auto import tqdm

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', None)


def _c(gc_=True):
    for k, v in list(globals().items()):
        if not (
            isinstance(v, ModuleType)
            or (callable(v) and (len(k) > 1 or isinstance(v, LambdaType)))
            or k in ['Counter', 'deepcopy', 'Path', 'pprint', 'ic', 'In', 'Out']
            or k.startswith('_')
            or k.isupper()
        ):
            del globals()[k]
    if gc_:
        return gc.collect()


def init_hv():
    hv.extension('bokeh')
