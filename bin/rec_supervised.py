# %%
import math
import tempfile
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, OrderedDict, Union, Literal
from copy import deepcopy
from pathlib import Path
from statistics import mean
from tqdm.auto import tqdm
from permutations import gen_permutations_class

import rtdl
import torch
import torch.nn as nn
import torch.nn.functional as F

import zero
import lib


@dataclass
class Config:
    @dataclass
    class Data:
        path: str
        T: lib.Transformations = field(default_factory=lib.Transformations)
        T_cache: bool = False

    @dataclass
    class Bins:
        count: int
        value: Literal['ratio', 'one'] = 'ratio'
        tree: Optional[dict[str, Any]] = None
        subsample: Union[None, int, float] = None

    @dataclass
    class Pretrain:
        corrupt_probability: float
        corrupt_strategy: Literal['resample', 'zero']
        d_hidden_head: int
        lr: float = None
        weight_decay: float = None
        loss_masked: bool = False
        patience: int = 2
        n_iterations: int = 100_000
        validate_every: int = 10_000
        replace_strategy: Literal['shuffle', 'target'] = 'shuffle'
        use_target: bool=False
        early_stop_type: Literal['finetune', 'pretrain'] = 'finetune'
        supervised_weight: int = 0.5
        predict_residual: bool=False

    @dataclass
    class Model:
        # Backbone hyperparams
        kind: str
        config: dict[str, Any] = field(default_factory=dict)
        default: bool = True
        checkpoint: Union[str, None] = None
        # Encoding hyperparams
        num_embedding_arch: list[str] = field(default_factory=list)
        d_num_embedding: Optional[int] = None
        d_cat_embedding: Union[None, int, Literal['d_num_embedding']] = None
        positional_encoding: Optional[lib.PeriodicEmbeddingOptions] = None

    @dataclass
    class Training:
        batch_size: int
        lr: float
        weight_decay: float
        optimizer: str = 'AdamW'
        patience: Optional[int] = 16
        n_epochs: Union[int, float] = float('inf')
        eval_batch_size: int = 8192

    seed: int
    data: Data
    model: Model
    training: Training
    pretrain: Pretrain
    bins: Optional[Bins] = None

    def __post_init__(self):
        if self.bins:
            self.bins = Config.Bins(**self.bins)
        if self.model.positional_encoding:
            self.model.positional_encoding = lib.PeriodicEmbeddingOptions(**self.model.positional_encoding)
        if self.pretrain.lr is None:
            self.pretrain.lr = self.training.lr
        if self.pretrain.weight_decay is None:
            self.pretrain.weight_decay = self.training.weight_decay

        if self.model.kind == "transformer":
            if 'ffn_d_hidden_factor' in self.model.config:
                lib.replace_factor_with_value(
                    self.model.config, 'ffn_d_hidden', self.model.d_num_embedding, (0.5, 8.0),
                )
        elif self.model.kind == "resnet":
            lib.replace_factor_with_value(
                self.model.config, 'd_hidden', self.model.config['d_main'], (1.0, 8.0)
            )


C, output, report = lib.start(Config)

zero.improve_reproducibility(C.seed)
device = lib.get_device()
D = lib.build_dataset(C.data.path, C.data.T, C.data.T_cache)

report['epoch_size'] = math.ceil(D.size('train') / C.training.batch_size)

X_num, X_cat, Y = lib.prepare_tensors(D, device=device)

if C.bins:
    bins_store = lib.construct_bins(D, C, device)
else:
    bins_store = None


def gen_masks(X, perm):
    masks = torch.empty_like(X).bernoulli(p=C.pretrain.corrupt_probability).bool()
    new_masks = masks & (X != X[perm, torch.arange(X.shape[1], device=X.device)])
    return new_masks

gen_permutations = gen_permutations_class(C.pretrain.replace_strategy, X_num, X_cat, Y, D)

permutations = {part: gen_permutations(part) for part in ['train', 'val', 'test']}
permutations_num = {part: permutations[part][0] for part in ['train', 'val', 'test']}
permutations_cat = {part: permutations[part][1] for part in ['train', 'val', 'test']}
masks_num = {part: gen_masks(X_num[part], permutations_num[part]) for part in ['train', 'val', 'test']}

if X_cat:
    masks_cat = {part: gen_masks(X_cat[part], permutations_cat[part]) for part in ['train', 'val', 'test']}


class Head(nn.Module):
    "2 Layer MLP projection head"

    def __init__(
        self,
        *,
        d_in: int,
        d_hidden: int,
        n_num_features: int,
        category_sizes: Optional[list[int]] = None,
    ):
        super().__init__()
        self.n_num_features = n_num_features
        self.first = nn.Linear(d_in, d_hidden)
        self.num_out = lib.NLinear(n_num_features, d_hidden, 1)
        if category_sizes:
            self.cat_out = nn.ModuleList([nn.Linear(d_hidden, o) for o in category_sizes])
        else:
            self.cat_out = None

    def forward(self, h: torch.Tensor, x_num_init: torch.Tensor=None):
        h = F.relu(self.first(h))

        if self.cat_out:
            cat_out = [f(h) for f in self.cat_out]
        else:
            cat_out = None

        if h.ndim == 2:
            h = h.unsqueeze(1).repeat(1, self.n_num_features, 1)

        num_out = self.num_out(h).squeeze()
        if C.pretrain.predict_residual:
            return num_out + x_num_init, cat_out
        return num_out, cat_out


class CLSHead(nn.Module):
    "2 Layer MLP projection head"

    def __init__(
        self,
        *,
        d_in: int,
        d_hidden: int,
        n_num_features: int,
        category_sizes: Optional[list[int]] = None,
    ):
        super().__init__()
        self.n_num_features = n_num_features
        self.first = nn.Linear(d_in, d_hidden)
        self.num_out = nn.Linear(d_hidden, 1)
        if category_sizes:
            self.cat_out = nn.ModuleList([nn.Linear(d_hidden, o) for o in category_sizes])
        else:
            self.cat_out = None
        self.category_sizes = category_sizes

    def forward(self, x: torch.Tensor):
        x = x[:, 1:]
        x = F.relu(self.first(x))

        if self.cat_out:
            x_cat = x[:, -len(self.category_sizes ):, :]
            cat_out = [f(x_cat[:, i]) for i, f in enumerate(self.cat_out)]
        else:
            cat_out = None

        x_num = x
        if self.category_sizes:  
            x_num = x[:, :-len(self.category_sizes), :]
        num_out = self.num_out(x_num).squeeze()
        return num_out, cat_out


class TargetHead(Head):
    def __init__(self,
                 *,
                 d_in: int,
                 d_hidden: int,
                 n_num_features: int,
                 numeric_target: bool,
                 category_sizes: Optional[list[int]] = None,
                 n_classes: int=None):
        n_y_bias = 1 if (numeric_target or n_classes == 2) else n_classes
        super().__init__(d_in=d_in+n_y_bias,
                         d_hidden=d_hidden,
                         n_num_features=n_num_features,
                         category_sizes=category_sizes)
        self.numeric_target = numeric_target
        self.n_classes = n_classes
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.numeric_target or self.n_classes == 2:
            y = y[:, None]
        else:
            y = y.long()
            y = torch.nn.functional.one_hot(y, self.n_classes)
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.first(x))

        if self.cat_out:
            cat_out = [f(x) for f in self.cat_out]
        else:
            cat_out = None

        if x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, self.n_num_features, 1)

        num_out = self.num_out(x).squeeze()

        return num_out, cat_out


class PretrainModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Input modules
        d_cat_embedding = C.model.d_cat_embedding
        d_num_embedding = C.model.d_num_embedding

        self.category_sizes = D.get_category_sizes('train')
        if (
            self.category_sizes and
            (C.model.kind == "transformer" or C.model.d_cat_embedding == "d_num_embedding")
        ):
            d_cat_embedding = C.model.d_num_embedding

        if d_num_embedding:
            self.num_embeddings = lib.NumEmbeddings(
                C.model.num_embedding_arch,
                D.n_num_features,
                d_num_embedding,
                d_feature=bins_store.n_bins if bins_store else None,
                periodic_embedding_options=C.model.positional_encoding,
            )
            d_in_num = D.n_num_features * C.model.d_num_embedding
        else:
            self.num_embeddings = None
            d_in_num = bins_store.n_bins if bins_store else D.n_num_features

        if d_cat_embedding:
            self.cat_embeddings = rtdl.CategoricalFeatureTokenizer(
                self.category_sizes, d_cat_embedding, True, 'uniform'
            )
            d_in_cat = d_cat_embedding * D.n_cat_features
        else:
            self.cat_embeddings = None
            d_in_cat = sum(self.category_sizes)

        d_in = d_in_num + d_in_cat
        print(f"Model: Built embeddings flattened input dim: {d_in}")

        # Backbones
        self.cls_token = None
        if C.model.kind == "mlp":
            self.backbone = lib.MLP(d_in, C.model.config, headless=False, d_out=D.nn_output_dim)
        elif C.model.kind == "resnet":
            self.backbone = lib.ResNet(d_in, C.model.config)
        elif C.model.kind == "transformer":
            if C.model.default:
                default_config = rtdl.FTTransformer.get_default_transformer_config(n_blocks=3)
                C.model.config = default_config | C.model.config
            else:
                baseline_config = rtdl.FTTransformer.get_baseline_transformer_subconfig()
                C.model.config = baseline_config | C.model.config
            C.model.config['d_token'] = C.model.d_num_embedding
            self.backbone = lib.Transformer(C.model.config, headless=False, d_out=D.nn_output_dim)
            self.cls_token = rtdl.CLSToken(self.backbone.d, 'uniform')

        if C.pretrain.use_target:
            self.head = TargetHead(
                d_in=self.backbone.d,
                n_num_features=D.n_num_features,
                category_sizes=D.get_category_sizes("train"),
                d_hidden=C.pretrain.d_hidden_head,
                numeric_target=D.is_regression,
                n_classes=D.n_classes if not D.is_binclass else 2
            )
        else:
            if C.model.kind == "transformer":
                self.head = CLSHead(
                    d_in=self.backbone.d,
                    n_num_features=D.n_num_features,
                    category_sizes=D.get_category_sizes("train"),
                    d_hidden=C.pretrain.d_hidden_head,
                )
            else:
                self.head = Head(
                    d_in=self.backbone.d,
                    n_num_features=D.n_num_features,
                    category_sizes=D.get_category_sizes("train"),
                    d_hidden=C.pretrain.d_hidden_head,
                )

    def forward(self, x_num, x_cat, y=None):
        if self.num_embeddings:
            x_num = self.num_embeddings(x_num)
        if self.cat_embeddings is not None:
            assert x_cat is not None
            x_cat = self.cat_embeddings(x_cat)
        elif x_cat is not None:
            x_cat = torch.concat(
                [
                    nn.functional.one_hot(x_cat[:, i], category_size)
                    for i, category_size in enumerate(self.category_sizes)
                ],
                1,
            )

        x = torch.cat(
            [x_.flatten(1,2) if x_.ndim == 3 and self.backbone.FLAT else x_ for x_ in [x_num, x_cat] if x_ is not None],
            dim=1
        )
        if self.cls_token:
            x = self.cls_token(x)
        if C.model.kind == "mlp":
            h = self.backbone.mlp.blocks(x)
            if C.pretrain.use_target:
                return self.head(h, y), self.backbone.mlp.head(h)
            return self.head(h), self.backbone.mlp.head(h)
        elif C.model.kind == "transformer":
            for layer in self.backbone.transformer.blocks:
                layer: nn.ModuleDict = layer

                x_residual = self.backbone.transformer._start_residual(layer, 'attention', x)
                x_residual, _ = layer['attention'](
                    x_residual,
                    x_residual,
                    *self.backbone.transformer._get_kv_compressions(layer),
                )
                x = self.backbone.transformer._end_residual(layer, 'attention', x, x_residual)

                x_residual = self.backbone.transformer._start_residual(layer, 'ffn', x)
                x_residual = layer['ffn'](x_residual)
                x = self.backbone.transformer._end_residual(layer, 'ffn', x, x_residual)
                x = layer['output'](x)
            h = x
            if C.pretrain.use_target:
                return self.head(h, y), self.backbone.transformer.head(h)
            return self.head(h), self.backbone.transformer.head(h)


model = PretrainModel().to(device)
supervised_loss_fn = lib.get_loss_fn(D.task_type)

if torch.cuda.device_count() > 1:
    print('Using nn.DataParallel')
    model = nn.DataParallel(model)  # type: ignore[code]
report['n_parameters'] = lib.get_n_parameters(model)

optimizer_config = asdict(C.training)
optimizer = lib.make_optimizer(
    optimizer_config, lib.split_parameters_by_weight_decay(model)
)

stream = zero.Stream(
    zero.data.IndexLoader(D.size('train'), C.training.batch_size, True, device=device, drop_last=True)
)
progress = zero.ProgressTracker(C.pretrain.patience)
training_log = {}
eval_batch_size = C.training.eval_batch_size
chunk_size = None


def print_iteration_info():
    print(f'\n>>> Iteration {stream.iteration} | {timer} | {output}')

    print(
        ' | '.join(
            f'{k} = {v}'
            for k, v in {
                'lr': lib.get_lr(optimizer),
                'batch_size': C.training.batch_size,  # type: ignore[code]
                'chunk_size': chunk_size,
                'epoch_size': report['epoch_size'],
                'n_parameters': report['n_parameters'],
            }.items()
        )
    )


@torch.inference_mode()
def evaluate(parts):
    global eval_batch_size
    model.eval()
    metrics = {}

    for part in parts:
        pretrain_losses = []
        supervised_losses = []
        while eval_batch_size:
            try:
                for idx in zero.data.IndexLoader(
                    D.size(part), eval_batch_size, False, device=device
                ):
                    (pred_num, pred_cat), pred_y = apply_model(part, idx)
                    target_num = X_num[part][idx]
                    mask_num = masks_num[part][idx]
                    if pred_cat:
                        target_cat = X_cat[part][idx]
                        mask_cat = masks_cat[part][idx]
                    else:
                        target_cat = None
                        mask_cat = None
                    pretrain_losses.append(pretrain_loss_fn(pred_num, pred_cat, target_num, target_cat, mask_num, mask_cat).item())
                    supervised_losses.append(supervised_loss_fn(pred_y.squeeze(-1), Y[part][idx]).item())

            except RuntimeError as err:
                if not lib.is_oom_exception(err):
                    raise
                eval_batch_size //= 2
                print('New eval batch size:', eval_batch_size)
                report['eval_batch_size'] = eval_batch_size
            else:
                break
        if not eval_batch_size:
            RuntimeError('Not enough memory even for eval_batch_size=1')

        metrics[part] = {"pretrain_component": mean(pretrain_losses),
                         "supervised_component": mean(supervised_losses),
                         "pretrain_loss": 2 * ((1 - C.pretrain.supervised_weight) * mean(pretrain_losses) + C.pretrain.supervised_weight  * mean(supervised_losses))}

    zero.hardware.free_memory()
    return metrics


def finetune():
    "Run validation on latest checkpoint"
    config = asdict(deepcopy(C))
    del config["pretrain"]
    if C.pretrain.early_stop_type == 'pretrain':
        model.load_state_dict(torch.load(output/f"checkpoint.pt")['model'])
    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        torch.save(
            {
                "model": model.state_dict(),
                "bins_store": bins_store,
            },
            dir_/"checkpoint.pt"
        )
        config["model"]["checkpoint"] = str(dir_/f"checkpoint.pt")

        out = dir_ / f'finetune_{stream.iteration}'
        config_path = out.with_suffix('.toml')
        lib.dump_config(config, config_path)
        python = Path('/miniconda3/envs/main/bin/python')
        subprocess.run(
            [
                str(python) if python.exists() else "python",
                lib.get_path("bin/finetune.py"),
                str(config_path),
            ],
            check=True,
        )
        report = lib.load_report(out)
        predictions = lib.load_predictions(out)
        finetune_state_dict = torch.load(out/"checkpoint.pt")["model"]
        torch.save(finetune_state_dict, output/"finetune_checkpoint.pt")
        training_log = torch.load(out/"checkpoint.pt")["training_log"]
        zero.hardware.free_memory()
        return report['metrics'], training_log, predictions


def save_checkpoint():
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'stream': stream.state_dict(),
            'random_state': zero.random.get_state(),
            **{
                x: globals()[x]
                for x in [
                    'progress',
                    'report',
                    'timer',
                    'training_log',
                ]
            },
        },
        output/f"checkpoint.pt"
    )
    lib.dump_report(report, output)
    lib.backup_output(output)


def apply_model(part, idx):
    if bins_store:
        perm_num = permutations_num[part][idx]
        bins_perm = torch.gather(bins_store.bins[part], 0, perm_num)
        bin_values_perm = torch.gather(bins_store.bin_values[part], 0, perm_num)

        x_num_permuted = lib.bin_encoding(bins_perm, bin_values_perm, bins_store.n_bins, D.n_num_features)
        x_num = lib.bin_encoding(bins_store.bins[part][idx], bins_store.bin_values[part][idx], bins_store.n_bins, D.n_num_features)
    else:
        perm_num = permutations_num[part][idx]
        x_num_permuted = torch.gather(X_num[part], 0, perm_num)
        x_num = X_num[part][idx]

    x_masks_num = masks_num[part][idx]
    x_num[x_masks_num] = x_num_permuted[x_masks_num]

    if X_cat:
        perm_cat = permutations_cat[part][idx]
        x_cat_permuted = torch.gather(X_cat[part], 0, perm_cat)
        x_cat = X_cat[part][idx]
        x_masks_cat = masks_cat[part][idx]
        x_cat[x_masks_cat] = x_cat_permuted[x_masks_cat]
    else:
        x_cat = None
    if C.pretrain.use_target:
        y = Y[part][idx]
        return model(x_num, x_cat, y)
    return model(x_num, x_cat)


def step_fn(idx):
    (pred_num, pred_cat), pred_y = apply_model("train", idx)
    mask_num = masks_num["train"][idx]
    target_num = X_num["train"][idx]
    if pred_cat:
        target_cat = X_cat["train"][idx]
        mask_cat = masks_cat["train"][idx]
    else:
        target_cat = None
        mask_cat = None
    return pred_num, pred_cat, pred_y.squeeze(-1), target_num, target_cat, Y['train'][idx], mask_num, mask_cat


def pretrain_loss_fn(pred_num, pred_cat, target_num, target_cat, mask_num, mask_cat):
    if C.pretrain.loss_masked:
        loss = F.mse_loss(pred_num[mask_num], target_num[mask_num])
    else:
        loss = F.mse_loss(pred_num, target_num)

    if pred_cat:
        for i, p in enumerate(pred_cat):
            if C.pretrain.loss_masked:
                column_mask = mask_cat[:, i]
                loss += F.cross_entropy(p[column_mask], target_cat[column_mask, i].long())
            else:
                loss += F.cross_entropy(p, target_cat[:, i].long())
    return loss


def loss_fn(pred_num, pred_cat, pred_y, target_num, target_cat, target_y, mask_num, mask_cat):
    return 2 * ((1 - C.pretrain.supervised_weight) * pretrain_loss_fn(pred_num, pred_cat, target_num, target_cat, mask_num, mask_cat) + C.pretrain.supervised_weight * supervised_loss_fn(pred_y, target_y))


def stream_iterations(n_iterations):
    it = 0
    while it < n_iterations:
        yield stream.next()
        it += 1


timer = lib.Timer.launch()

for batch_idx in tqdm(stream_iterations(C.pretrain.n_iterations)):
    model.train()

    loss, new_chunk_size = lib.train_with_auto_virtual_batch(
        optimizer,
        loss_fn,
        step_fn,
        batch_idx,
        chunk_size or C.training.batch_size,
    )

    if new_chunk_size and new_chunk_size < (chunk_size or C.training.batch_size):
        report['chunk_size'] = chunk_size = new_chunk_size
        print('New chunk size:', chunk_size)

    if stream.iteration % report["epoch_size"] == 0:
        permutations["train"] = gen_permutations("train")
        permutations_num["train"] = permutations["train"][0]
        permutations_cat["train"] = permutations["train"][1]
        masks_num["train"] = gen_masks(X_num["train"], permutations_num["train"])

        if X_cat:
            masks_cat["train"] = gen_masks(X_cat["train"], permutations_cat["train"])

        print_iteration_info()
        print(f"[train_loss] {loss.item():.3f}")

    if stream.iteration % C.pretrain.validate_every == 0:
        pretrain_metrics = evaluate(['train', 'val', 'test'])
        metrics = {part: pretrain_metrics[part] for part in pretrain_metrics.keys()}
        if C.pretrain.early_stop_type == 'finetune':
            finetune_metrics, finetune_logs, predictions = finetune()
            for part in finetune_metrics.keys():
                metrics[part].update(finetune_metrics[part])

            report.setdefault("metrics", OrderedDict()).setdefault("iteration_scores", OrderedDict())[stream.iteration] = {
                part: {
                    "score": metrics[part]["score"],
                    "pretrain_loss": metrics[part]["pretrain_loss"],
                } for part in metrics.keys()
            }
            progress.update(metrics["val"]["score"])
            training_log.setdefault("finetune_logs", OrderedDict())[stream.iteration] = finetune_logs
            print(lib.format_scores(metrics), end=' ')
        else:
            report.setdefault("metrics", OrderedDict()).setdefault("iteration_scores", OrderedDict())[stream.iteration] = {
                part: {
                    "pretrain_loss": metrics[part]["pretrain_loss"],
                } for part in metrics.keys()
            }
            progress.update(-1 * metrics["val"]["pretrain_loss"])

        lib.update_training_log(training_log, {"time": timer()}, metrics)

        print_iteration_info()
        print(f"[val_loss] {metrics['val']['pretrain_loss']:.3f}")

        if progress.success:
            print("New best pretrain iteration!")
            report["best_iteration"] = stream.iteration
            if C.pretrain.early_stop_type == 'finetune':
                for part in metrics.keys():
                    report["metrics"][part] = {"score": metrics[part]["score"]}
                lib.dump_report(report, output)
                lib.dump_predictions(predictions, output)
            save_checkpoint()

        elif progress.fail:
            print("Early stopping!")
            break

if C.pretrain.early_stop_type == 'pretrain':
    finetune_metrics, finetune_logs, predictions = finetune()
    for part in metrics.keys():
        report["metrics"][part] = {"score": finetune_metrics[part]["score"]}
    lib.dump_report(report, output)
    lib.dump_predictions(predictions, output)
report['time'] = str(timer)
lib.finish(output, report)
