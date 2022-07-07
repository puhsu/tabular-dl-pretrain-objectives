import statistics
from typing import Any, Callable, Optional, Union, Literal, cast
from dataclasses import dataclass

import numpy as np
import lib
import rtdl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import zero
from torch import Tensor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .util import TaskType

@dataclass
class Bins:
    bins: dict[str, Tensor]
    bin_values: dict[str, Tensor]
    n_bins: int
    bin_edges: list[np.array]


def construct_bins(D: lib.Dataset, C, device):
    print('\nRunning bin-based encoding...')
    assert D.X_num is not None
    assert C.bins is not None
    bin_edges = []
    _bins = {x: [] for x in D.X_num}
    _bin_values = {x: [] for x in D.X_num}
    rng = np.random.default_rng(C.seed)
    for feature_idx in tqdm.trange(D.n_num_features):
        train_column = D.X_num['train'][:, feature_idx].copy()
        if C.bins.subsample is not None:
            subsample_n = (
                C.bins["subsample"]
                if isinstance(C.bins.subsample, int)
                else int(C.bins.subsample * D.size('train'))
            )
            subsample_idx = rng.choice(len(train_column), subsample_n, replace=False)
            train_column = train_column[subsample_idx]
        else:
            subsample_idx = None

        if C.bins.tree is not None:
            _y = D.y['train']
            if subsample_idx is not None:
                _y = _y[subsample_idx]
            tree = (
                (DecisionTreeRegressor if D.is_regression else DecisionTreeClassifier)(
                    max_leaf_nodes=C.bins.count, **C.bins.tree
                )
                .fit(train_column.reshape(-1, 1), D.y['train'])
                .tree_
            )
            del _y
            tree_thresholds = []
            for node_id in range(tree.node_count):
                # the following condition is True only for split nodes
                # See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    tree_thresholds.append(tree.threshold[node_id])
            tree_thresholds.append(train_column.min())
            tree_thresholds.append(train_column.max())
            bin_edges.append(np.array(sorted(set(tree_thresholds))))
        else:
            feature_n_bins = min(C.bins.count, len(np.unique(train_column)))
            quantiles = np.linspace(
                0.0, 1.0, feature_n_bins + 1
            )  # includes 0.0 and 1.0
            bin_edges.append(np.unique(np.quantile(train_column, quantiles)))

        for part in D.X_num:
            _bins[part].append(
                np.digitize(
                    D.X_num[part][:, feature_idx],
                    np.r_[-np.inf, bin_edges[feature_idx][1:-1], np.inf],
                ).astype(np.int32)
                - 1
            )
            if C.bins.value == 'one':
                _bin_values[part].append(np.ones_like(D.X_num[part][:, feature_idx]))
            else:
                assert C.bins.value == 'ratio'
                if bin_edges[feature_idx].shape[0] == 1:
                    feature_bin_sizes = np.array([1])
                else:
                    feature_bin_sizes = (
                        bin_edges[feature_idx][1:] - bin_edges[feature_idx][:-1]
                    )
                part_feature_bins = _bins[part][feature_idx]
                _bin_values[part].append(
                    (
                        D.X_num[part][:, feature_idx]
                        - bin_edges[feature_idx][part_feature_bins]
                    )
                    / feature_bin_sizes[part_feature_bins]
                )

    n_bins = max(map(len, bin_edges)) - 1

    bins = {
        k: torch.as_tensor(np.stack(v, axis=1), dtype=torch.int64, device=device)
        for k, v in _bins.items()
    }
    del _bins
    bin_values = {
        k: torch.as_tensor(np.stack(v, axis=1), dtype=torch.float32, device=device)
        for k, v in _bin_values.items()
    }
    del _bin_values

    return Bins(
        bins,
        bin_values,
        n_bins,
        bin_edges,

    )


def bin_encoding(
    batch_bins: Tensor, batch_bin_values: Tensor,
    n_bins: int, n_num_features: int,
):
    device = batch_bins.device
    bs = batch_bins.size(0)

    bin_mask_ = torch.eye(n_bins, device=device)[batch_bins]
    x = bin_mask_ * batch_bin_values[..., None]
    previous_bins_mask = torch.arange(n_bins, device=device)[None, None].repeat(
        bs, n_num_features, 1
    ) < batch_bins.reshape(bs, n_num_features, 1)
    x[previous_bins_mask] = 1.0
    return x

# Models

class NLinear(nn.Module):
    def __init__(self, n: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n, d_in, d_out))
        self.bias = nn.Parameter(torch.Tensor(n, d_out))
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                self.bias[i] = layer.bias

    def forward(self, x):
        # x - b,n,d_in
        # x*w - b,n,d_in,d_out ->sum(-2) -> b,n,d_out
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        x = x + self.bias[None]
        return x


def cos_sin(x: Tensor) -> Tensor:
    return torch.cat([torch.cos(x), torch.sin(x)], -1)


@dataclass
class PeriodicEmbeddingOptions:
    n: int  # the output size is 2 * n
    sigma: float
    trainable: bool
    initialization: Literal['log-linear', 'normal']


class PeriodicEmbedding(nn.Module):
    def __init__(self, n_features: int, options: PeriodicEmbeddingOptions) -> None:
        super().__init__()
        if options.initialization == 'log-linear':
            coefficients = options.sigma ** (torch.arange(options.n) / options.n)
            coefficients = coefficients[None].repeat(n_features, 1)
        else:
            assert options.initialization == 'normal'
            coefficients = torch.normal(0.0, options.sigma, (n_features, options.n))
        if options.trainable:
            self.coefficients = nn.Parameter(coefficients)
        else:
            self.register_buffer('coefficients', coefficients)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        return cos_sin(2 * torch.pi * self.coefficients[None] * x[..., None])


class NumEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_arch: list[str],
        n_features: int,
        d_embedding: int,
        d_feature: Optional[int],
        periodic_embedding_options: Optional[PeriodicEmbeddingOptions],
    ) -> None:
        super().__init__()
        assert set(embedding_arch) <= {'relu', 'linear', 'shared_linear', 'positional'}
        assert embedding_arch

        if embedding_arch[0] == 'linear':
            assert periodic_embedding_options is None
            layers = [
                rtdl.NumericalFeatureTokenizer(n_features, d_embedding, True, 'uniform')
                if d_feature is None
                else NLinear(n_features, d_feature, d_embedding)
            ]
            d_current = d_embedding
        elif embedding_arch[0] == "positional":
            assert d_feature is None
            layers = [
                lib.PeriodicEmbedding(n_features, periodic_embedding_options)
            ]
            d_current = periodic_embedding_options.n * 2
        else:
            raise ValueError("Wrong first embedding layer")

        for x in embedding_arch[1:]:
            layers.append(
                nn.ReLU()
                if x == 'relu'
                else NLinear(n_features, d_current, d_embedding)
                if x == 'linear'
                else nn.Linear(d_current, d_embedding)
                if x == 'shared_linear'
                else None
            )
            d_current = d_embedding
            assert layers[-1] is not None
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Head(nn.Module):
    def __init__(
        self,
        n_features: Optional[int],
        d_in: int,
        d_hidden: int,
        d_out: int,
        flat_model=True,
    ) -> None:
        super().__init__()

        self.flat_model = flat_model

        layers = [
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        ]

        if n_features:
            if flat_model: layers.append(
                zero.nn.Lambda(lambda x: x.unsqueeze(1).repeat(1,n_features,1))
            )
            layers.append(NLinear(n_features, d_hidden, d_out))
        else:
            layers.append(nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return x

# Backbones:

class MLP(nn.Module):
    FLAT = True

    def __init__(self, d_in, config, headless=True, d_out=1):
        super().__init__()
        self.d = config["d_layers"][-1]
        self.headless = headless
        self.mlp = rtdl.MLP.make_baseline(
            d_in=d_in,
            **config,
            d_out=d_out,
        )
        if self.headless:
            del self.mlp.head # No heads in pretraining

    def forward(self, x):
        if self.headless:
            return self.mlp.blocks(x)

        return self.mlp(x)


class ResNet(nn.Module):
    FLAT = True

    def __init__(self, d_in, config, headless=True, d_out=1):
        super().__init__()
        self.d = config["d_main"]
        self.headless = headless
        self.resnet = rtdl.ResNet.make_baseline(
            d_in=d_in,
            **config,
            d_out=d_out,
        )
        if self.headless:
            del self.resnet.head # No heads in pretraining

    def forward(self, x):
        if self.headless:
            x = self.resnet.first_layer(x)
            return self.resnet.blocks(x)

        return self.resnet(x)


class Transformer(nn.Module):
    FLAT = False

    def __init__(self, config, headless=True, d_out=1):
        super().__init__()
        self.d = config["d_token"]
        self.headless = headless
        self.transformer = rtdl.Transformer(**config, d_out=d_out)
        if self.headless:
            del self.transformer.head

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3

        if not self.headless:
            return self.transformer(x)

        transformer = self.transformer

        for layer in transformer.blocks:
            layer: nn.ModuleDict = layer

            x_residual = transformer._start_residual(layer, 'attention', x)
            x_residual, _ = layer['attention'](
                x_residual,
                x_residual,
                *transformer._get_kv_compressions(layer),
            )
            x = transformer._end_residual(layer, 'attention', x, x_residual)

            x_residual = transformer._start_residual(layer, 'ffn', x)
            x_residual = layer['ffn'](x_residual)
            x = transformer._end_residual(layer, 'ffn', x, x_residual)
            x = layer['output'](x)

        return x


def make_cat_tokenizer(
    category_sizes: list[int], d_cat_token: Optional[int]
) -> tuple[int, Optional[rtdl.CategoricalFeatureTokenizer]]:
    if category_sizes:
        assert d_cat_token is not None
        tokenizer = rtdl.CategoricalFeatureTokenizer(
            category_sizes, d_cat_token, False, 'uniform'
        )
        return tokenizer.n_tokens * tokenizer.d_token, tokenizer
    else:
        assert d_cat_token is None
        return 0, None


def concat_num_cat(
    x_num: Optional[Tensor], x_cat: Optional[Tensor], cat_tokenizer: Optional[nn.Module]
):
    assert (
        x_num is not None or x_cat is not None
    ), 'At least one of the inputs must be presented.'

    x = []
    if x_num is not None:
        assert x_num.ndim == 2, 'The numerical input must have two dimensions.'
        x.append(x_num)
    if x_cat is None:
        assert (
            cat_tokenizer is None
        ), 'If `x_cat` is None, then `cat_input_module` also must be None.'
    else:
        assert (
            cat_tokenizer is not None
        ), 'If `x_cat` is not None, then `cat_input_module` also must not be None.'
        x.append(cat_tokenizer(x_cat).flatten(1, -1))
    return x[0] if len(x) == 1 else torch.cat(x, dim=1)


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


def get_loss_fn(task_type: TaskType) -> Callable[..., Tensor]:
    return (
        F.binary_cross_entropy_with_logits
        if task_type == TaskType.BINCLASS
        else F.cross_entropy
        if task_type == TaskType.MULTICLASS
        else F.mse_loss
    )


def default_zero_weight_decay_condition(module_name, module, parameter_name, parameter):
    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            rtdl.CLSToken,
            rtdl.NumericalFeatureTokenizer,
            rtdl.CategoricalFeatureTokenizer,
        ),
    )


def split_parameters_by_weight_decay(
    model: nn.Module, zero_weight_decay_condition=default_zero_weight_decay_condition,
    do_not_finetune_condition=None,
) -> list[dict[str, Any]]:
    parameters_info = {}
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters():
            full_parameter_name = (
                f'{module_name}.{parameter_name}' if module_name else parameter_name
            )

            if do_not_finetune_condition and do_not_finetune_condition(module_name, module, parameter_name, parameter):
                continue

            parameters_info.setdefault(full_parameter_name, ([], parameter))[0].append(
                zero_weight_decay_condition(
                    module_name, module, parameter_name, parameter
                )
            )
    params_with_wd = {'params': []}
    params_without_wd = {'params': [], 'weight_decay': 0.0}
    for full_parameter_name, (results, parameter) in parameters_info.items():
        (params_without_wd if any(results) else params_with_wd)['params'].append(
            parameter
        )
    return [params_with_wd, params_without_wd]


def make_optimizer(
    config: dict[str, Any],
    parameter_groups,
) -> optim.Optimizer:
    if config['optimizer'] == 'FT-Transformer-default':
        return optim.AdamW(parameter_groups, lr=1e-4, weight_decay=1e-5)
    return getattr(optim, config['optimizer'])(
        parameter_groups,
        **{x: config[x] for x in ['lr', 'weight_decay', 'momentum'] if x in config},
    )


def get_lr(optimizer: optim.Optimizer) -> float:
    return next(iter(optimizer.param_groups))['lr']


def get_ft_transformer_param_groups(
    model: Union[rtdl.FTTransformer, nn.DataParallel]
) -> list[dict[str, Any]]:
    return (
        model.optimization_param_groups()
        if isinstance(model, rtdl.FTTransformer)
        else model.module.optimization_param_groups()  # type: ignore[code]
    )


def is_oom_exception(err: RuntimeError) -> bool:
    return any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


def train_with_auto_virtual_batch(
    optimizer,
    loss_fn,
    step,
    batch,
    chunk_size: int,
) -> tuple[Tensor, int]:
    batch_size = len(batch)
    random_state = zero.random.get_state()
    while chunk_size != 0:
        try:
            zero.random.set_state(random_state)
            optimizer.zero_grad()
            if batch_size <= chunk_size:
                loss = loss_fn(*step(batch))
                loss.backward()
            else:
                loss = None
                for chunk in zero.iter_batches(batch, chunk_size):
                    chunk_loss = loss_fn(*step(chunk))
                    chunk_loss = chunk_loss * (len(chunk) / batch_size)
                    chunk_loss.backward()
                    if loss is None:
                        loss = chunk_loss.detach()
                    else:
                        loss += chunk_loss.detach()
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            chunk_size //= 2
        else:
            break
    if not chunk_size:
        raise RuntimeError('Not enough memory even for batch_size=1')
    optimizer.step()
    return cast(Tensor, loss), chunk_size


def process_epoch_losses(losses: list[Tensor]) -> tuple[list[float], float]:
    losses_ = torch.stack(losses).tolist()
    return losses_, statistics.mean(losses_)
