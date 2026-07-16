import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
try:
    from src.utils.rich_progress import progress, write
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress, write

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import configure_torch_fast_math, regression_metrics, resolve_amp_dtype, save_json, set_seed
from emulator_bench.dataset import CachedMPEKDataset
from emulator_bench.modeling import CachedMPEKRegressor


def make_loader(dataset, args, shuffle: bool):
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": False,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    amp_dtype,
    train: bool,
    target_mean: float,
    target_std: float,
    desc: str = "",
    no_progress: bool = False,
):
    model.train(train)
    losses, true_values, pred_values = [], [], []
    criterion = torch.nn.MSELoss()
    iterator = progress(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True, disable=no_progress)
    for step, batch in enumerate(iterator, start=1):
        protein = batch["protein"].to(device, non_blocking=True)
        ligand = batch["ligand"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(protein, ligand)
            loss = criterion(pred, target)
        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                if args_clip_grad := getattr(run_epoch, "clip_grad", 0.0):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args_clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args_clip_grad := getattr(run_epoch, "clip_grad", 0.0):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args_clip_grad)
                optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        raw_true = batch["raw_target"].numpy().reshape(-1)
        raw_pred = pred.detach().float().cpu().numpy().reshape(-1) * target_std + target_mean
        true_values.append(raw_true)
        pred_values.append(raw_pred)
        if step == 1 or step % 25 == 0:
            iterator.set_postfix(loss=f"{loss_value:.5f}", avg_loss=f"{float(np.mean(losses)):.5f}")
    y_true = np.concatenate(true_values) if true_values else np.array([])
    y_pred = np.concatenate(pred_values) if pred_values else np.array([])
    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, y_true, y_pred


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, args, target_mean, target_std):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_metric": float(best_metric),
            "args": vars(args),
            "target_mean": float(target_mean),
            "target_std": float(target_std),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"]), float(checkpoint.get("best_metric", float("inf")))


def write_predictions(path: Path, y_true, y_pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)


def write_metrics(path: Path, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train one cached MPEK split.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task", type=str, default="unknown")
    parser.add_argument("--split_group", type=str, default="unknown")
    parser.add_argument("--split_name", type=str, default="unknown")
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--expert_dim", type=int, default=768)
    parser.add_argument("--expert_layers", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--ple_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tower_layers", type=int, default=3)
    parser.add_argument("--tower_hidden", type=int, default=128)
    parser.add_argument("--tower_dropout", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--limit_rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    configure_torch_fast_math()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    final_metrics = out_dir / "final_results_test.csv"
    if final_metrics.exists() and not args.overwrite:
        print(f"Completed run already exists, skipping: {final_metrics}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "run_config.json", vars(args))

    train_ds = CachedMPEKDataset(
        args.train_path,
        args.embeddings_dir,
        args.sequence_col,
        args.smiles_col,
        args.target_col,
        limit_rows=args.limit_rows,
        preload=args.preload,
    )
    val_ds = CachedMPEKDataset(
        args.val_path,
        args.embeddings_dir,
        args.sequence_col,
        args.smiles_col,
        args.target_col,
        target_mean=train_ds.target_mean,
        target_std=train_ds.target_std,
        limit_rows=args.limit_rows,
        preload=args.preload,
    )
    test_ds = CachedMPEKDataset(
        args.test_path,
        args.embeddings_dir,
        args.sequence_col,
        args.smiles_col,
        args.target_col,
        target_mean=train_ds.target_mean,
        target_std=train_ds.target_std,
        limit_rows=args.limit_rows,
        preload=args.preload,
    )
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)
    test_loader = make_loader(test_ds, args, shuffle=False)

    device = torch.device(args.device)
    amp_dtype, precision = resolve_amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    model = CachedMPEKRegressor(
        expert_dim=args.expert_dim,
        expert_layers=args.expert_layers,
        num_experts=args.num_experts,
        ple_layers=args.ple_layers,
        dropout=args.dropout,
        tower_layers=args.tower_layers,
        tower_hidden=args.tower_hidden,
        tower_dropout=args.tower_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    run_epoch.clip_grad = args.clip_grad

    latest_path = out_dir / "checkpoint_latest.pt"
    best_path = out_dir / "checkpoint_best.pt"
    start_epoch, best_rmse = 0, float("inf")
    if latest_path.exists() and not args.overwrite:
        start_epoch, best_rmse = load_checkpoint(latest_path, model, optimizer, scheduler, device)
        print(f"Resuming from epoch {start_epoch} with best validation RMSE {best_rmse:.6f}")
    print(f"Training on {device} with precision {precision}")
    run_label = f"{args.task}/{args.split_group}/{args.split_name}/seed{args.seed}"

    history = []
    started = time.time()
    epoch_iterator = progress(
        range(start_epoch + 1, args.epochs + 1),
        desc=f"{run_label} epochs",
        unit="epoch",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for epoch in epoch_iterator:
        train_metrics, _train_true, _train_pred = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_dtype,
            True,
            train_ds.target_mean,
            train_ds.target_std,
            desc=f"{run_label} train {epoch}/{args.epochs}",
            no_progress=args.no_progress,
        )
        val_metrics, _val_true, _val_pred = run_epoch(
            model,
            val_loader,
            optimizer,
            None,
            device,
            amp_dtype,
            False,
            train_ds.target_mean,
            train_ds.target_std,
            desc=f"{run_label} val {epoch}/{args.epochs}",
            no_progress=args.no_progress,
        )
        scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "epoch_metrics.csv", index=False)
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_rmse, args, train_ds.target_mean, train_ds.target_std)
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_rmse, args, train_ds.target_mean, train_ds.target_std)
        if not args.no_progress:
            epoch_iterator.set_postfix(train_rmse=f"{train_metrics['rmse']:.5f}", val_rmse=f"{val_metrics['rmse']:.5f}", best=f"{best_rmse:.5f}")
        print(f"epoch={epoch} train_rmse={train_metrics['rmse']:.6f} val_rmse={val_metrics['rmse']:.6f} best={best_rmse:.6f}", flush=True)

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    for split_name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        metrics, y_true, y_pred = run_epoch(
            model,
            loader,
            None,
            None,
            device,
            amp_dtype,
            False,
            train_ds.target_mean,
            train_ds.target_std,
            desc=f"{run_label} final {split_name}",
            no_progress=args.no_progress,
        )
        metrics.update(
            {
                "task": args.task,
                "split_group": args.split_group,
                "split_name": args.split_name,
                "seed": args.seed,
                "n": int(len(y_true)),
                "elapsed_seconds": time.time() - started,
            }
        )
        write_metrics(out_dir / f"final_results_{split_name}.csv", metrics)
        write_predictions(out_dir / f"predictions_{split_name}.csv", y_true, y_pred)


if __name__ == "__main__":
    main()
