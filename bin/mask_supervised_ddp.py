import math
import tempfile
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, OrderedDict, Union, Literal
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm
from permutations import gen_permutations_class

import rtdl
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

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
        corrupt_strategy: Literal['resample']
        d_hidden_head: int
        lr: float = None
        weight_decay: float = None
        patience: int = 2
        n_iterations: int = 100_000
        validate_every: int = 10_000
        replace_strategy: Literal['shuffle', 'target'] = 'shuffle'
        use_target: bool=False
        early_stop_type: Literal['finetune', 'pretrain'] = 'finetune'
        supervised_weight: int = 0.5

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
        positional_encoding: Optional[lib.PeriodicEmbeddingOptions] = None
        d_cat_embedding: Union[None, int, Literal['d_num_embedding']] = None

    @dataclass
    class Training:
        batch_size: int
        lr: float
        weight_decay: float
        optimizer: str = 'AdamW'
        patience: Optional[int] = 16
        n_epochs: Union[int, float] = float('inf')
        eval_batch_size: int = 1024

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


class Head(nn.Module):
    "2 Layer MLP projection head"

    def __init__(
        self,
        *,
        d_in: int,
        d_hidden: int,
        n_features: int,
    ):
        super().__init__()
        self.n_features = n_features
        self.first = nn.Linear(d_in, d_hidden)
        self.out = lib.NLinear(n_features, d_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.first(x))

        if x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, self.n_features, 1)

        return self.out(x).squeeze()


class CLSHead(nn.Module):
    "2 Layer MLP projection head"

    def __init__(
        self,
        *,
        d_in: int,
        d_hidden: int,
        use_target: False
    ):
        super().__init__()
        self.first = nn.Linear(d_in, d_hidden)
        self.out = nn.Linear(d_hidden, 1)
        self.use_target = use_target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 1:]
        if self.use_target:
            x = x[:, :-1]

        x = self.out(F.relu(self.first(x))).squeeze(2)
        return x


class TargetHead(nn.Module):
    "2 Layer MLP projection head"

    def __init__(
        self,
        *,
        d_in: int,
        d_hidden: int,
        n_features: int,
        numeric_target: bool,
        n_classes: int=None
    ):
        super().__init__()
        self.n_features = n_features
        n_y_bias = 1 if (numeric_target or n_classes == 2) else n_classes
        self.first = nn.Linear(d_in + n_y_bias, d_hidden)
        self.out = lib.NLinear(n_features, d_hidden, 1)
        self.numeric_target = numeric_target
        self.n_classes = n_classes
        # if numeric_target:
        #     self.y_extractor = nn.Sequential(nn.Linear(1, d_in), nn.ReLU(), nn.Linear(d_in, d_in))
        # else:
        #     self.y_extractor =  nn.Sequential(nn.Embedding(n_classes, d_in), nn.Linear(d_in, d_in))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.numeric_target or self.n_classes == 2:
            y = y[:, None]
        else:
            y = y.long()
            y = torch.nn.functional.one_hot(y, self.n_classes)

        x = torch.cat([x, y], dim=1)
        x = F.relu(self.first(x))

        if x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, self.n_features, 1)

        return self.out(x).squeeze()


class PretrainModel(nn.Module):
    def __init__(self, C: Config, D: lib.Dataset, bins_store: lib.Bins):
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
            self.backbone = lib.Transformer(C.model.config)
            self.cls_token = rtdl.CLSToken(self.backbone.d, 'uniform')
        if C.pretrain.use_target:
            self.head = TargetHead(
                d_in=self.backbone.d,
                n_features=D.n_num_features + D.n_cat_features,
                d_hidden=C.pretrain.d_hidden_head,
                numeric_target=D.is_regression,
                n_classes=D.n_classes if not D.is_binclass else 2
            )
        else:
            if C.model.kind == "transformer":
                self.head = CLSHead(d_in=self.backbone.d,
                                    d_hidden=C.pretrain.d_hidden_head,
                                    use_target=insert_pre_target)
            else:
                self.head = Head(
                    d_in=self.backbone.d,
                    n_features=D.n_num_features + D.n_cat_features,
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
        inputs = [x_num, x_cat]

        x = torch.cat(
            [x_.flatten(1,2) if x_.ndim == 3 and self.backbone.FLAT else x_ for x_ in inputs if x_ is not None],
            dim=1
        )
        if self.cls_token:
            x = self.cls_token(x)

        h = self.backbone.mlp.blocks(x)
        if isinstance(self.head, TargetHead):
            assert y is not None
            return self.head(h, y), self.backbone.mlp.head(h)
        return self.head(h), self.backbone.mlp.head(h)

def train(
    C: Config,
    output: Path,
    report: dict,
):
    "Training loop"

    zero.improve_reproducibility(C.seed, one_cuda_seed=True)
    device = lib.get_device()
    D = lib.build_dataset(C.data.path, C.data.T, C.data.T_cache)

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

    model = PretrainModel(C, D, bins_store).to(device)
    supervised_loss_fn = lib.get_loss_fn(D.task_type)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[lib.get_rank()])
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    report['n_parameters'] = lib.get_n_parameters(model)
    report['epoch_size'] = math.ceil(D.size('train') / C.training.batch_size)

    optimizer_config = asdict(C.training)
    optimizer = lib.make_optimizer(
        optimizer_config, lib.split_parameters_by_weight_decay(model)
    )

    progress = zero.ProgressTracker(C.pretrain.patience)
    training_log = {}
    eval_batch_size = C.training.eval_batch_size
    chunk_size = None

    stream = zero.Stream(
        lib.IndexLoader(D.size('train'), C.training.batch_size // lib.get_world_size(), True, ddp=dist.is_initialized())
    )

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

    def save_checkpoint():
        if lib.get_rank() == 0:
            if dist.is_initialized():
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(
                {
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'stream': stream.state_dict(),
                    'random_state': zero.random.get_state(),
                    'progress': progress,
                    'report': report,
                    'timer': timer,
                    'training_log': training_log,
                },
                output/'checkpoint.pt',
            )
            lib.dump_report(report, output)
            lib.backup_output(output)

        lib.synchronize()


    @torch.inference_mode()
    def evaluate(parts):
        model.eval()
        metrics = {}

        for part in parts:
            loader = lib.IndexLoader(
                D.size(part),
                batch_size=eval_batch_size,
                shuffle=False,
                ddp=dist.is_initialized(),
            )
            i = []
            p_s = []
            p_p = []

            for idx in loader:
                i.append(idx)
                cur_pretrain_predictions, cur_supervised_predictions = apply_model(part, idx)
                p_s.append(cur_supervised_predictions)
                p_p.append(cur_pretrain_predictions)

            i = torch.cat(i)
            p_s = torch.cat(p_s)
            p_p = torch.cat(p_p)

            if dist.is_initialized():
                world_size = lib.get_world_size()
                indexes_l = list(torch.empty(world_size, i.size(0), dtype=i.dtype, device=i.device).unbind(0))
                sup_predictions_l = list(torch.empty(world_size, p_s.size(0), p_s.size(1), dtype=p_s.dtype, device=p_s.device).unbind(0))
                pr_predictions_l = list(torch.empty(world_size, p_p.size(0), p_p.size(1), dtype=p_p.dtype, device=p_p.device).unbind(0))
                dist.all_gather(indexes_l, i)
                dist.all_gather(sup_predictions_l, p_s)
                dist.all_gather(pr_predictions_l, p_p)

                indexes = torch.cat(indexes_l)
                sup_predictions_ = torch.cat(sup_predictions_l)
                sup_predictions = torch.empty(D.size(part), sup_predictions_.size(1), dtype=sup_predictions_.dtype, device=sup_predictions_.device)
                sup_predictions.index_copy_(0, indexes, sup_predictions_)

                pr_predictions_ = torch.cat(pr_predictions_l)
                pr_predictions = torch.empty(D.size(part), pr_predictions_.size(1), dtype=pr_predictions_.dtype, device=pr_predictions_.device)
                pr_predictions.index_copy_(0, indexes, pr_predictions_)
            else:
                sup_predictions = p_s
                pr_predictions = p_p

            if X_cat:
                masks = torch.cat([masks_num[part], masks_cat[part]], dim=1)
            else:
                masks = masks_num[part]
            hard_predictions = torch.zeros_like(pr_predictions, dtype=torch.long)
            hard_predictions[pr_predictions > 0] = 1
            features_accuracy = (hard_predictions.bool() == masks).sum(0) / hard_predictions.shape[0]
            pretrain_loss = pretrain_loss_fn(pr_predictions, masks).item()
            supervised_loss = supervised_loss_fn(sup_predictions.squeeze(-1), Y[part]).item()
            metrics[part] = {"pretrain_loss": 2 * ((1 - C.pretrain.supervised_weight) * pretrain_loss + C.pretrain.supervised_weight * supervised_loss),
                            "pretrain_component": pretrain_loss,
                            "supervised_component": supervised_loss,
                            "features_accuracy" : features_accuracy.tolist()}

        zero.hardware.free_memory()
        return metrics


    def finetune():
        "Run finetune on saved checkpoint"
        config = asdict(deepcopy(C))
        del config["pretrain"]
        if C.pretrain.early_stop_type == 'pretrain':
            state_dict = torch.load(output/f"checkpoint.pt")['model']
        else:
            if dist.is_initialized():
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

        with tempfile.TemporaryDirectory() as dir_:
            dir_ = Path(dir_)

            torch.save(
                {
                    "model": state_dict,
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
                    lib.get_path("bin/finetune_ddp.py"),
                    str(config_path),
                ],
                check=True,
            )
            report = lib.load_report(out)
            training_log = torch.load(out/"checkpoint.pt")["training_log"]
            predictions = lib.load_predictions(out)
            zero.hardware.free_memory()
            return report['metrics'], training_log, predictions


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


    def get_masks(part, idx):
        if X_cat:
            return torch.cat([masks_num[part][idx], masks_cat[part][idx]], dim=1)

        return masks_num[part][idx]


    def step_fn(idx):
        pretrain_predictions, supervised_predictions = apply_model("train", idx)
        return pretrain_predictions, supervised_predictions.squeeze(-1), get_masks("train", idx), Y['train'][idx]

    def pretrain_loss_fn(predictions, masks):
        return F.binary_cross_entropy_with_logits(predictions, masks.float())

    def loss_fn(pretrain_predictions, supervised_predictions, masks, target):
        pretrain_loss = pretrain_loss_fn(pretrain_predictions, masks)
        supervised_loss = supervised_loss_fn(supervised_predictions, target)
        loss = 2 * ((1 - C.pretrain.supervised_weight) * pretrain_loss + C.pretrain.supervised_weight * supervised_loss)
        return loss


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
                        "pretrain_component": metrics[part]["pretrain_component"],
                        "supervised_component": metrics[part]["supervised_component"],
                        "features_accuracy": metrics[part]["features_accuracy"]
                    } for part in metrics.keys()
                }
                progress.update(metrics["val"]["score"])
                training_log.setdefault("finetune_logs", OrderedDict())[stream.iteration] = finetune_logs
                print(lib.format_scores(metrics), end=' ')
            else:
                report.setdefault("metrics", OrderedDict()).setdefault("iteration_scores", OrderedDict())[stream.iteration] = {
                    part: {
                        "pretrain_loss": metrics[part]["pretrain_loss"],
                        "pretrain_component": metrics[part]["pretrain_component"],
                        "supervised_component": metrics[part]["supervised_component"],
                        "features_accuracy": metrics[part]["features_accuracy"]
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
                lib.synchronize()

            elif progress.fail:
                print("Early stopping!")
                break

    if lib.get_rank() == 0:
        if C.pretrain.early_stop_type == 'pretrain':
            finetune_metrics, finetune_logs, predictions = finetune()
            for part in metrics.keys():
                report["metrics"][part] = {"score": finetune_metrics[part]["score"]}
            lib.dump_report(report, output)
            lib.dump_predictions(predictions, output)
        report['time'] = str(timer)
        lib.finish(output, report)


def main():
    C, output, report = lib.start(Config)

    if torch.cuda.device_count() > 1:
        print("Launching DDP training")
        port = lib.find_free_port()
        dist_url = f"tcp://127.0.0.1:{port}"
        world_size = torch.cuda.device_count()
        train_args = (C, output, report)

        mp.spawn(
            dist_worker,
            nprocs=world_size,
            args=(
                train,
                world_size,
                dist_url,
                train_args,
            )
        )
    else:
        train(C, output, report)


def dist_worker(rank, fn, world_size, dist_url, args):

    dist.init_process_group(
        backend="NCCL",
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(rank)
    lib.setup_distributed_logs(rank == 0)

    lib.synchronize()
    fn(*args)


if __name__ == "__main__":
    main()
