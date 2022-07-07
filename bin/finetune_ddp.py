# %%
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union, Literal
from pathlib import Path

import rtdl
import torch
import torch.nn as nn
import zero

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

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
        n_epochs: Union[int, float] = math.inf
        eval_batch_size: int = 1024

    seed: int
    data: Data
    model: Model
    training: Training
    bins: Optional[Bins] = None

    def __post_init__(self):
        if self.bins:
            self.bins = Config.Bins(**self.bins)
        if self.model.positional_encoding:
            self.model.positional_encoding = lib.PeriodicEmbeddingOptions(**self.model.positional_encoding)

        if self.model.kind == "transformer":
            self.model.config.setdefault('last_layer_query_idx', [-1])

            if 'ffn_d_hidden_factor' in self.model.config:
                lib.replace_factor_with_value(
                    self.model.config,
                    'ffn_d_hidden',
                    self.model.d_num_embedding,
                    (0.5, 8.0),
                )
        elif self.model.kind == "resnet":
            lib.replace_factor_with_value(
                self.model.config, 'd_hidden', self.model.config['d_main'], (1.0, 8.0)
            )


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
            self.backbone = lib.ResNet(d_in, C.model.config, headless=False, d_out=D.nn_output_dim)
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
        if self.cls_token:
            x = self.cls_token(x)
        return self.backbone(x)


def train(
    C: Config,
    output: Path,
    report: dict,
):
    "Training loop"

    zero.improve_reproducibility(C.seed)
    device = lib.get_device()
    D = lib.build_dataset(C.data.path, C.data.T, C.data.T_cache)

    X_num, X_cat, Y = lib.prepare_tensors(D, device=device)

    ckpt = dict()

    if C.model.checkpoint is not None:
        ckpt: dict = torch.load(C.model.checkpoint, map_location="cpu")

    if C.bins:
        bins_store = ckpt.get("bins_store") or lib.construct_bins(D, C, device)
        bins_store.bins = {p: v.to(device) for p,v in bins_store.bins.items()}
        bins_store.bin_values = {p: v.to(device) for p,v in bins_store.bin_values.items()}
    else:
        bins_store = None

    model = PretrainModel(C, D, bins_store)
    loss_fn = lib.get_loss_fn(D.task_type)

    if C.model.checkpoint is not None:
        incompatible_keys = model.load_state_dict(ckpt["model"], strict=False)
        print('Loaded checkpoint. Incompatible keys:')
        print(incompatible_keys)

    model = model.to(device)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[lib.get_rank()])
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    report['n_parameters'] = lib.get_n_parameters(model)
    report['prediction_type'] = None if D.is_regression else 'logits'
    report['epoch_size'] = math.ceil(D.size('train') / C.training.batch_size)

    optimizer = lib.make_optimizer(
        asdict(C.training), lib.split_parameters_by_weight_decay(model),
    )

    chunk_size = None
    eval_batch_size = C.training.eval_batch_size
    progress = zero.ProgressTracker(C.training.patience)
    checkpoint_path = output / 'checkpoint.pt'
    training_log = {}

    if lib.get_rank() == 0:
        lib.dump_pickle(D.y_info, output / 'y_info.pickle')

    stream = zero.Stream(
        lib.IndexLoader(D.size('train'), C.training.batch_size // lib.get_world_size(), True, ddp=dist.is_initialized())
    )

    lib.synchronize()

    # Utils

    def print_epoch_info():
        if lib.get_rank() != 0: return
        print(f'\n>>> Epoch {stream.epoch} | {timer} | {output}')
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
                checkpoint_path,
            )
            lib.dump_report(report, output)
            lib.backup_output(output)

        lib.synchronize()


    def apply_model(part, idx):
        if bins_store:
            bins = bins_store.bins[part][idx]
            bin_values = bins_store.bin_values[part][idx]
            x_num = lib.bin_encoding(bins, bin_values, bins_store.n_bins, D.n_num_features)
        else:
            x_num = X_num[part][idx]

        if X_cat:
            x_cat = X_cat[part][idx]
        else:
            x_cat = None

        return model(x_num, x_cat).squeeze(-1)


    @torch.inference_mode()
    def evaluate(parts):
        model.eval()
        predictions = {}
        for part in parts:
            loader = lib.IndexLoader(
                D.size(part),
                batch_size=eval_batch_size,
                shuffle=False,
                ddp=dist.is_initialized(),
            )
            i = []
            p = []

            for idx in loader:
                i.append(idx)
                p.append(apply_model(part, idx))

            i = torch.cat(i)
            p = torch.cat(p)

            if dist.is_initialized():
                world_size = lib.get_world_size()
                indexes_l = list(torch.empty(world_size, i.size(0), dtype=i.dtype, device=i.device).unbind(0))
                predictions_l = list(torch.empty(world_size, *p.shape, dtype=p.dtype, device=p.device).unbind(0))

                dist.all_gather(indexes_l, i)
                dist.all_gather(predictions_l, p)

                indexes = torch.cat(indexes_l)
                predictions_ = torch.cat(predictions_l)
                predictions[part] = torch.empty(D.size(part), *p.shape[1:], dtype=predictions_.dtype, device=predictions_.device)
                predictions[part].index_copy_(0, indexes, predictions_)
                predictions[part] = predictions[part].cpu().numpy()
            else:
                predictions[part] = p.cpu().numpy()
            lib.synchronize()
            try:
                metrics = D.calculate_metrics(predictions, report['prediction_type'])
            except ValueError:
                print("Diverged! error in metrics")
                metrics = {part: {'score': -float('inf')} for part in parts}

        return metrics, predictions


    # Run train
    timer = lib.Timer.launch()
    if lib.get_rank() == 0:
        pbar = zero._stream._DEFAULT_PROGRESS_BAR_CONFIG
    else:
        pbar = None

    for epoch in stream.epochs(C.training.n_epochs, progress_bar_config=pbar):
        print_epoch_info()

        model.train()
        epoch_losses = []

        for batch_idx in epoch:
            loss, new_chunk_size = lib.train_with_auto_virtual_batch(
                optimizer,
                loss_fn,
                lambda x: (apply_model('train', x), Y['train'][x]),
                batch_idx,
                chunk_size or C.training.batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.training.batch_size):
                report['chunk_size'] = chunk_size = new_chunk_size
                print('New chunk size:', chunk_size)

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        metrics, predictions = evaluate(['val', 'test'])

        lib.update_training_log(
            training_log,
            {
                'train_loss': epoch_losses,
                'mean_train_loss': mean_loss,
                'time': timer(),
            },
            metrics,
        )
        print(f'\n{lib.format_scores(metrics)} [loss] {mean_loss:.3f}')

        progress.update(metrics['val']['score'])
        if progress.success:
            print('New best epoch!')
            report['best_epoch'] = stream.epoch
            report['metrics'] = metrics
            save_checkpoint()
            if lib.get_rank() == 0:
                lib.dump_predictions(predictions, output)

        elif progress.fail:
            break

    lib.synchronize()
    if dist.is_initialized():
        model.module.load_state_dict(torch.load(checkpoint_path)['model'])
    else:
        model.load_state_dict(torch.load(checkpoint_path)['model'])
    lib.synchronize()

    report['metrics'], predictions = evaluate(['train', 'val', 'test'])
    report['time'] = str(timer)
    save_checkpoint()
    if lib.get_rank() == 0:
        lib.dump_predictions(predictions, output)

    if lib.get_rank() == 0:
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
