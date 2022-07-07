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
import numpy as np

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
        finetune_every: int = 200
        replace_strategy: Literal['shuffle'] = 'shuffle'

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
                    self.model.config, 'ffn_d_hidden', self.model.config['d_token'], (0.5, 8.0),
                )
        elif self.model.kind == "resnet":
            lib.replace_factor_with_value(
                self.model.config, 'd_hidden', self.model.config['d_main'], (1.0, 8.0)
            )

class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


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
        self.out = nn.Linear(d_hidden, d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(F.relu(self.first(x)))

        return x


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
        if C.model.kind == "mlp":
            self.backbone = lib.MLP(d_in, C.model.config)
        elif C.model.kind == "resnet":
            self.backbone = lib.ResNet(d_in, C.model.config)
        elif C.model.kind == "transformer":
            if C.model.default:
                default_config = rtdl.FTTransformer.get_default_transformer_config(n_blocks=3)
                C.model.config = default_config | C.model.config
            else:
                baseline_config = rtdl.FTTransformer.get_baseline_transformer_subconfig()
                C.model.config = baseline_config | C.model.config

            self.backbone = lib.Transformer(C.model.config)
            self.cls_token = rtdl.CLSToken(self.backbone.d)

        self.head = Head(
            d_in=self.backbone.d,
            n_features=D.n_num_features + D.n_cat_features,
            d_hidden=C.pretrain.d_hidden_head,
        )

    def forward(self, x_num, x_cat):
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

        h = self.backbone(x)
        return self.head(h)


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

    def binarize_target(y):
        device = y.device
        y_numpy = y.detach().cpu().numpy()
        bins = np.searchsorted(np.histogram_bin_edges(y_numpy, bins='fd'), y_numpy)
        return torch.tensor(bins, dtype=torch.long).to(device)

    gen_permutations = gen_permutations_class(C.pretrain.replace_strategy, X_num, X_cat, Y, D)

    permutations = {part: gen_permutations(part) for part in ['train', 'val', 'test']}
    permutations_num = {part: permutations[part][0] for part in ['train', 'val', 'test']}
    permutations_cat = {part: permutations[part][1] for part in ['train', 'val', 'test']}
    masks_num = {part: gen_masks(X_num[part], permutations_num[part]) for part in ['train', 'val', 'test']}

    if X_cat:
        masks_cat = {part: gen_masks(X_cat[part], permutations_cat[part]) for part in ['train', 'val', 'test']}

    Y_binarized = Y
    if D.is_regression:
        Y_binarized = {part: binarize_target(Y_binarized[part]) for part in ['train', 'val', 'test']}

    model = PretrainModel(C, D, bins_store).to(device)

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[lib.get_rank()])
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    report['n_parameters'] = lib.get_n_parameters(model)
    report['epoch_size'] = math.ceil(D.size('train') / C.training.batch_size)

    optimizer_config = asdict(C.training)
    optimizer_config["lr"] = C.pretrain.lr
    optimizer = lib.make_optimizer(
        optimizer_config, lib.split_parameters_by_weight_decay(model)
    )

    progress = zero.ProgressTracker(C.pretrain.patience)
    training_log = {}
    eval_batch_size = C.training.eval_batch_size
    chunk_size = None

    stream = zero.Stream(
        lib.IndexLoader(D.size('train'), C.training.batch_size // lib.get_world_size(), True, ddp=dist.is_initialized(), drop_last=True)
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
            loss = 0
            for idx in loader:
                original_view = apply_model(part, idx, False)
                corrupted_view = apply_model(part, idx, True)
                loss += loss_fn(original_view, corrupted_view, Y_binarized[part][idx])

            loss /= len(loader)
            if dist.is_initialized():
                loss /= dist.get_world_size()
                dist.all_reduce(loss)

            metrics[part] = {"pretrain_loss": loss.item()}
        zero.hardware.free_memory()
        return metrics


    def finetune():
        "Run finetune on saved checkpoint"
        config = asdict(deepcopy(C))
        del config["pretrain"]
        # model.load_state_dict(torch.load(output/f"checkpoint.pt")['model'])
        with tempfile.TemporaryDirectory() as dir_:
            dir_ = Path(dir_)
            state_dict = torch.load(output/f"checkpoint.pt")['model']

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


    def apply_model(part, idx, corrupt=False):
        if corrupt:
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
        else:
            if bins_store:
                x_num = lib.bin_encoding(bins_store.bins[part][idx], bins_store.bin_values[part][idx], bins_store.n_bins, D.n_num_features)
            else:
                x_num = X_num[part][idx]
            x_cat = None
            if X_cat:
                x_cat = X_cat[part][idx]

        return model(x_num, x_cat)

    def step_fn(idx):
        original_view = apply_model("train", idx, False)
        corrupted_view = apply_model("train", idx, True)
        return original_view, corrupted_view, Y_binarized['train'][idx]

    def loss_fn(original_view, corrupted_view, y):
        # supervised contrastive loss: https://arxiv.org/pdf/2004.11362.pdf
        original_view = F.normalize(original_view, dim=1)
        corrupted_view = F.normalize(corrupted_view, dim=1)

        if dist.is_initialized():
            original_view = torch.cat(GatherLayer.apply(original_view), dim=0)
            corrupted_view = torch.cat(GatherLayer.apply(corrupted_view), dim=0)
            y = torch.cat(GatherLayer.apply(y))

        sim = F.log_softmax(original_view @ corrupted_view.T, dim=1)
        mask = torch.logical_or(torch.eye(sim.shape[0]).bool().to(sim.device), (y[None] == y[:, None]))

        loss = (-1 * (mask * sim).sum(1) / mask.sum(1)[:, None]).mean()
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

        if stream.iteration % C.pretrain.finetune_every == 0:
            pretrain_metrics = evaluate(['train', 'val', 'test'])
            # finetune_metrics, finetune_logs, predictions = finetune()
            metrics = {part: pretrain_metrics[part] for part in pretrain_metrics.keys()}
            report.setdefault("metrics", OrderedDict()).setdefault("iteration_scores", OrderedDict())[stream.iteration] = {
                part: {
                    "pretrain_loss": metrics[part]["pretrain_loss"],
                } for part in metrics.keys()
            }
            progress.update(-1 * metrics["val"]["pretrain_loss"])

            # training_log.setdefault("finetune_logs", OrderedDict())[stream.iteration] = finetune_logs
            lib.update_training_log(training_log, {"time": timer()}, metrics)

            print_iteration_info()
            print(f"[val_loss] {metrics['val']['pretrain_loss']:.3f}")

            if progress.success:
                print("New best pretrain iteration!")
                report["best_iteration"] = stream.iteration
                save_checkpoint()
                lib.synchronize()

            elif progress.fail:
                print("Early stopping!")
                break

    if lib.get_rank() == 0:
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
