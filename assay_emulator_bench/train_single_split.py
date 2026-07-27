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

from assay_emulator_bench.common import configure_torch_fast_math, regression_metrics, resolve_amp_dtype, save_json, set_seed
from assay_emulator_bench.dataset import CachedMPEKDataset
from assay_emulator_bench.evaluation import checkpoint_rmse, evaluate_prediction_rows
from assay_emulator_bench.modeling import CachedMPEKRegressor


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
    collect_metadata: bool = False,
):
    model.train(train)
    losses, true_values, pred_values = [], [], []
    metadata = {}
    criterion = torch.nn.MSELoss()
    iterator = progress(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True, disable=no_progress)
    for step, batch in enumerate(iterator, start=1):
        protein = batch["protein"].to(device, non_blocking=True)
        ligand = batch["ligand"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        model_kwargs = {}
        if model.use_assay_conditions:
            model_kwargs = {
                "pH": batch["pH"].to(device, non_blocking=True),
                "temperature": batch["temperature_C"].to(
                    device, non_blocking=True
                ),
            }
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(protein, ligand, **model_kwargs)
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
        if collect_metadata:
            batch_metadata = {
                "row_index": batch["row_index"].numpy().reshape(-1).tolist(),
                "sequence": list(batch["sequence"]),
                "smiles": list(batch["smiles"]),
                "assay_mapping_status": list(batch["assay_mapping_status"]),
                "source_pH_imputed": batch["source_pH_imputed"]
                .numpy()
                .reshape(-1)
                .tolist(),
                "source_Temperature_imputed": batch[
                    "source_Temperature_imputed"
                ]
                .numpy()
                .reshape(-1)
                .tolist(),
                "pH_range_imputed": batch["pH_range_imputed"]
                .numpy()
                .reshape(-1)
                .tolist(),
                "Temperature_range_imputed": batch[
                    "Temperature_range_imputed"
                ]
                .numpy()
                .reshape(-1)
                .tolist(),
            }
            if "pH" in batch:
                batch_metadata["pH_input"] = (
                    batch["pH"].numpy().reshape(-1).tolist()
                )
                batch_metadata["Temperature_C_input"] = batch[
                    "temperature_C"
                ].numpy().reshape(-1).tolist()
            for key, values in batch_metadata.items():
                metadata.setdefault(key, []).extend(values)
        if step == 1 or step % 25 == 0:
            iterator.set_postfix(loss=f"{loss_value:.5f}", avg_loss=f"{float(np.mean(losses)):.5f}")
    y_true = np.concatenate(true_values) if true_values else np.array([])
    y_pred = np.concatenate(pred_values) if pred_values else np.array([])
    predictions = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    for key, values in metadata.items():
        predictions[key] = values
    metrics = regression_metrics(predictions["y_true"], predictions["y_pred"])
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, predictions


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


def load_checkpoint(path, model, optimizer, scheduler, device, args):
    checkpoint = torch.load(path, map_location=device)
    saved_args = checkpoint.get("args", {})
    compatibility_defaults = {
        "use_assay_conditions": False,
        "pair_reducer": "none",
    }
    for key, default in compatibility_defaults.items():
        saved = saved_args.get(key, default)
        current = getattr(args, key)
        if saved != current:
            raise RuntimeError(
                f"Checkpoint setting mismatch for {key}: saved={saved!r}, "
                f"current={current!r}"
            )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"]), float(checkpoint.get("best_metric", float("inf")))


def write_predictions(path: Path, predictions: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)


def write_predictions_parquet(path: Path, predictions: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.parquet")
    predictions.to_parquet(temporary, index=False)
    temporary.replace(path)


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
    parser.add_argument("--use_assay_conditions", action="store_true")
    parser.add_argument("--ph_col", type=str, default="pH")
    parser.add_argument("--temperature_col", type=str, default="Temperature")
    parser.add_argument(
        "--temperature_unit",
        choices=["kelvin", "celsius"],
        default="kelvin",
    )
    parser.add_argument(
        "--pair_reducer",
        choices=["none", "max"],
        default="none",
    )
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
        use_assay_conditions=args.use_assay_conditions,
        ph_col=args.ph_col,
        temperature_col=args.temperature_col,
        temperature_unit=args.temperature_unit,
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
        use_assay_conditions=args.use_assay_conditions,
        ph_col=args.ph_col,
        temperature_col=args.temperature_col,
        temperature_unit=args.temperature_unit,
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
        use_assay_conditions=args.use_assay_conditions,
        ph_col=args.ph_col,
        temperature_col=args.temperature_col,
        temperature_unit=args.temperature_unit,
    )
    save_json(
        out_dir / "condition_summary.json",
        {
            "use_assay_conditions": args.use_assay_conditions,
            "temperature_unit": args.temperature_unit,
            "train": train_ds.condition_diagnostics,
            "val": val_ds.condition_diagnostics,
            "test": test_ds.condition_diagnostics,
        },
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
        use_assay_conditions=args.use_assay_conditions,
        condition_device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    run_epoch.clip_grad = args.clip_grad

    latest_path = out_dir / "checkpoint_latest.pt"
    best_path = out_dir / "checkpoint_best.pt"
    start_epoch, best_rmse = 0, float("inf")
    if latest_path.exists() and not args.overwrite:
        start_epoch, best_rmse = load_checkpoint(
            latest_path, model, optimizer, scheduler, device, args
        )
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
        train_metrics, _train_predictions = run_epoch(
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
        val_row_metrics, val_predictions = run_epoch(
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
            collect_metadata=args.pair_reducer != "none",
        )
        _row_metrics, _val_pairs, val_pair_metrics = evaluate_prediction_rows(
            val_predictions, args.pair_reducer
        )
        selection_rmse = checkpoint_rmse(
            val_row_metrics, val_pair_metrics, args.pair_reducer
        )
        scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        if args.pair_reducer == "none":
            row.update(
                {f"val_{key}": value for key, value in val_row_metrics.items()}
            )
        else:
            row.update(
                {
                    f"val_assay_row_{key}": value
                    for key, value in val_row_metrics.items()
                }
            )
            row.update(
                {
                    f"val_pair_{key}": value
                    for key, value in val_pair_metrics.items()
                }
            )
            row["val_selection_rmse"] = selection_rmse
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "epoch_metrics.csv", index=False)
        if selection_rmse < best_rmse:
            best_rmse = selection_rmse
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_rmse, args, train_ds.target_mean, train_ds.target_std)
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_rmse, args, train_ds.target_mean, train_ds.target_std)
        if not args.no_progress:
            epoch_iterator.set_postfix(train_rmse=f"{train_metrics['rmse']:.5f}", val_rmse=f"{selection_rmse:.5f}", best=f"{best_rmse:.5f}")
        print(f"epoch={epoch} train_rmse={train_metrics['rmse']:.6f} val_rmse={selection_rmse:.6f} best={best_rmse:.6f}", flush=True)

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    for split_name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        row_metrics, predictions = run_epoch(
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
            collect_metadata=args.pair_reducer != "none",
        )
        shared_metadata = {
            "task": args.task,
            "split_group": args.split_group,
            "split_name": args.split_name,
            "seed": args.seed,
            "elapsed_seconds": time.time() - started,
            "use_assay_conditions": args.use_assay_conditions,
            "pair_reducer": args.pair_reducer,
        }
        if args.pair_reducer == "none":
            metrics = {
                **row_metrics,
                **shared_metadata,
                "evaluation_level": "row",
                "n": int(len(predictions)),
            }
            write_metrics(out_dir / f"final_results_{split_name}.csv", metrics)
            write_predictions(
                out_dir / f"predictions_{split_name}.csv",
                predictions[["y_true", "y_pred"]],
            )
            continue

        _row_metrics, pair_predictions, pair_metrics = evaluate_prediction_rows(
            predictions, args.pair_reducer
        )
        row_output_metrics = {
            **row_metrics,
            **shared_metadata,
            "evaluation_level": "assay_row",
            "n": int(len(predictions)),
        }
        pair_output_metrics = {
            **pair_metrics,
            **shared_metadata,
            "evaluation_level": "pair",
            "n": int(len(pair_predictions)),
        }
        write_metrics(
            out_dir / f"final_results_{split_name}_assay_rows.csv",
            row_output_metrics,
        )
        write_metrics(
            out_dir / f"final_results_{split_name}.csv",
            pair_output_metrics,
        )
        write_predictions_parquet(
            out_dir / f"predictions_{split_name}_assay_rows.parquet",
            predictions.sort_values("row_index").reset_index(drop=True),
        )
        write_predictions_parquet(
            out_dir / f"predictions_{split_name}_pairs.parquet",
            pair_predictions,
        )


if __name__ == "__main__":
    main()
