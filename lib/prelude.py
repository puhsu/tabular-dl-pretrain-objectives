# flake8: noqa
# type: ignore
import enum
from typing import Union

from . import env as _env
from .data import load_dataset_info
from .prelude_std import *
from .prelude_std import _c
from .util import Part, TaskType

XGBOOST = 'xgboost_'
LIGHTGBM = 'lightgbm_'
CATBOOST = 'catboost_'
MLP = 'mlp'
RESNET = 'resnet'
FTT = 'ft_transformer'

GESTURE = 'gesture'
CHURN = 'churn'
CHURN_ROC = 'churn_roc'
EYE = 'eye'
CALIFORNIA = 'california'
HOUSE = 'house'
ADULT = 'adult'
ADULT_ROC = 'adult_roc'
OTTO = 'otto'
OTTO_LL = 'otto_ll'
HIGGS_SMALL = 'higgs-small'
HIGGS_SMALL_ROC = 'higgs-small_roc'
FB_COMMENTS = 'fb-comments'
SANTANDER = 'santander'
SANTANDER_ROC = 'santander_roc'
COVTYPE = 'covtype'
MICROSOFT = 'microsoft'
NOMAO = 'nomao'
WEATHER_SMALL = 'weather-small'

SMALL_DATASETS = [
    GESTURE,
    # CHURN,
    CHURN_ROC,
    # EYE,
    CALIFORNIA,
    HOUSE,
    # ADULT,
    # OTTO,
    OTTO_LL,
    # HIGGS_SMALL,
    HIGGS_SMALL_ROC,
    FB_COMMENTS,
    ADULT_ROC,
]
BIG_DATASETS = [
    WEATHER_SMALL,
    COVTYPE,
    MICROSOFT,
]
DATASETS = SMALL_DATASETS + BIG_DATASETS



def show_datasets():
    df = pd.DataFrame.from_records(list(map(load_dataset_info, DATASETS)))
    df = df.sort_values('size')
    df = df.reset_index(drop=True)
    df = df[
        [
            'name',
            'train_size',
            'val_size',
            'test_size',
            'size',
            'n_features',
            'n_num_features',
            'n_cat_features',
            'task_type',
            'n_classes',
            'path',
        ]
    ]
    return df


def make_command(program: str, config: Union[str, Path], suffix: str = '') -> str:
    cmd = f'python bin/{program}.py {_env.get_relative_path(config)}'
    if suffix:
        cmd = cmd + ' ' + suffix
    return cmd


def append_command(commands: list[str], *args, **kwargs):
    commands.append(make_command(*args, **kwargs))


def dump_commands(commands, rewrite=False):
    with open(_env.PROJ / 'ygorishniy' / 'commands.txt', 'w' if rewrite else 'a') as f:
        f.write('\n')
        f.write('\n'.join(commands))
        f.write('\n')


NN_BATCH_SIZES = {}
NN_NORMALIZATIONS = dict.fromkeys(DATASETS, 'quantile') | {OTTO: None, OTTO_LL: None, EYE: 'standard'}

for x in show_datasets().itertuples():
    name = x.path.name
    if x.train_size < 20000:
        NN_BATCH_SIZES[name] = 128
    elif x.train_size < 50000:
        NN_BATCH_SIZES[name] = 256
    elif x.train_size < 200000:
        NN_BATCH_SIZES[name] = 512
    elif x.train_size < 2000000:
        NN_BATCH_SIZES[name] = 1024
    else:
        assert False
