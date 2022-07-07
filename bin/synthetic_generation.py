import dataclasses as dc
import typing as ty
from pathlib import Path

import numpy as np
import zero

import lib
from lib import ArrayDict
from tqdm import tqdm

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer


train_size = 50_000
n_features = 8

# %%
AnyDict = ty.Dict[str, ty.Any]


@dc.dataclass
class Dataset:
    X: ArrayDict
    y: ArrayDict
    info: AnyDict
    basename: dc.InitVar[str]
    fi: ty.List
    

    def __post_init__(self, basename):
        split = 0
        self.info.update(
            {
                'basename': basename,
                'split': split,
                'name': f"{basename}-{split}",
                'task_type': 'binclass',
                'n_num_features': self.X['train'].shape[1],
                'n_cat_features': 0,
                **{f"{k}_size": len(v) for k, v in self.X.items()},
                'fi': self.fi
            }
        )


def save_dataset(dataset: Dataset, path: Path):
    path.mkdir()
    lib.dump_json(dataset.info, path / 'info.json')
    for name, arrays in [('X_num', dataset.X), ('y', dataset.y)]:
        for part in ['train', 'val', 'test']:
            np.save(path / f'{name}_{part}.npy', arrays[part])


def generate_X(train_size: int, n_features: int, seed: int) -> lib.ArrayDict:
    rng = np.random.default_rng(seed)

    def generate(fraction):
        return rng.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.diag(np.zeros(n_features) + 0.5) + 0.5,
            size=(int(fraction * train_size),),
        )

    return {'train': generate(1.0), 'val': generate(0.2), 'test': generate(0.3)}


@dc.dataclass(frozen=True)
class ObliviousForestTargetConfig:
    n_trees: int
    tree_depth: int
    seed: int


def generate_oblivious_forest_target(X: ArrayDict, config: ObliviousForestTargetConfig,
                                     importances: ty.List):
    zero.improve_reproducibility(config.seed)

    rng = np.random.default_rng(config.seed)
    X_all = np.concatenate([X['train'], X['val'], X['test']])
    bounds = np.column_stack([X_all.min(0), X_all.max(0)])

    n_splits = config.n_trees * config.tree_depth
    tree_feature_indices = np.random.choice(range(8), p=importances, size=n_splits)
    tree_feature_indices = tree_feature_indices[:n_splits]

    tree_feature_indices = rng.permutation(tree_feature_indices)
    tree_feature_indices = tree_feature_indices.reshape(
        config.n_trees, config.tree_depth
    )
    tree_feature_indices = tree_feature_indices.tolist()
    trees = []
    for feature_indices in tqdm(tree_feature_indices):
        thresholds = []
        leaf_values = []
        for feature_index in feature_indices:
            threshold = rng.uniform(bounds[feature_index][0],bounds[feature_index][1] )
            thresholds.append(threshold)

        thresholds = np.array(thresholds)
        leaf_values = rng.standard_normal((2,) * config.tree_depth, dtype='float32')
        targets = {
            k: leaf_values[
                tuple((v[:, feature_indices] < thresholds[None]).astype('int64').T)
            ]
            for k, v in X.items()
        }

        trees.append((thresholds, leaf_values, targets))

    result = {}
    for part in X.keys():
        ph = np.zeros(X[part].shape[0])
        for tree in tqdm(trees):
            ph += tree[-1][part]
        result[part] = ph

    qt = QuantileTransformer(output_distribution='normal')
    result['train'] = qt.fit_transform(result['train'][:, None])[:, 0]
    result['val'] = qt.transform(result['val'][:, None])[:, 0]
    result['test'] = qt.transform(result['test'][:, None])[:, 0]
    result = {part: (result[part] > 0) for part in ['train', 'val', 'test']}
    return (
        result,
        {'tree_feature_indices': tree_feature_indices},
    )


def train_catboost(catboost, X_num, y):
    cb = catboost(thread_count=10)
    cb.fit(X_num['train'], y['train'], eval_set=(X_num['val'], y['val']))
    metric = roc_auc_score(y['test'], cb.predict_proba(X_num['test'])[:, 1])
    return metric, cb


X = generate_X(train_size, n_features, 0)
for d in range(50):
    synthetic_path = lib.PROJ / 'data' / 'synthetic' / f'sd{d}'
    seed = d
    synthetic_path_seed = synthetic_path

    config = ObliviousForestTargetConfig(10, 10, seed)
    p = np.random.dirichlet(np.ones(n_features))
    
    X_new = {part: X[part] for part in ['train', 'val', 'test']}
    Y_np, info = generate_oblivious_forest_target(X_new, config,  p)

    X = {part: X[part].astype('float32') for part in ['train','val', 'test']}
    Y = {part: Y_np[part].astype('int32') for part in ['train','val', 'test']}

    catboost = CatBoostClassifier
    metric_all, cb = train_catboost(catboost, X, Y)
    dataset = Dataset(
            X,
            Y,
            {},
            f"synthetic",
            list(cb.get_feature_importance() / 100)
        )
    save_dataset(dataset, synthetic_path_seed)