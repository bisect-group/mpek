import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import torch.nn.functional as F

from assay_emulator_bench.common import MODES, regression_metrics, resolve_amp_dtype, save_json, set_seed
from assay_emulator_bench.dataset import AssayDataset
from assay_emulator_bench.modeling import CachedMPEKRegressor
from MTLKcatKM.utils import ChildTuningAdamW
from src.utils.rich_progress import progress

EVAL_BATCH_SIZE = 4096


def features(dataset, index, device):
    return [t[index].to(device) for t in (dataset.protein, dataset.ligand, dataset.pH, dataset.temperature)]


@torch.no_grad()
def predict(model, dataset, device):
    model.eval()
    preds = [
        model(*features(dataset, slice(start, start + EVAL_BATCH_SIZE), device)).cpu()
        for start in range(0, len(dataset), EVAL_BATCH_SIZE)
    ]
    return torch.cat(preds).squeeze(1) * dataset.target_std + dataset.target_mean


def train_epoch(model, optimizer, inputs, target, batch_size, amp_dtype=None, scaler=None):
    """One shuffled pass over device-resident tensors; returns the dropout-mode predictions."""
    model.train()
    order = torch.randperm(len(target), device=target.device)
    train_pred = torch.zeros_like(target)
    for start in range(0, len(target) - batch_size + 1, batch_size):
        idx = order[start : start + batch_size]
        with torch.autocast(device_type=target.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(*(t[idx] for t in inputs))
        pred = pred.float()
        loss = F.mse_loss(pred, target[idx])
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        train_pred[idx] = pred.detach()
    return train_pred


def assay_and_pair_metrics(dataset, y_pred):
    """Metrics per assay row and per (sequence, smiles) pair, taking the maximum true and predicted value of each pair."""
    rows = pd.DataFrame({"pair_id": dataset.pair_id.numpy(), "y_true": dataset.raw_target.numpy(), "y_pred": y_pred.numpy()})
    pairs = rows.groupby("pair_id").max()
    metrics = regression_metrics(rows["y_true"], rows["y_pred"])
    metrics.update({f"pair_{k}": v for k, v in regression_metrics(pairs["y_true"], pairs["y_pred"]).items()})
    return metrics, rows, len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Train one MPEK kcat split on assay rows with or without pH and temperature, using the original MPEK settings.")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--split_group", type=str, default="unknown")
    parser.add_argument("--split_name", type=str, default="unknown")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--expert_dim", type=int, default=768)
    parser.add_argument("--expert_layers", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--ple_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--tower_layers", type=int, default=2)
    parser.add_argument("--tower_hidden", type=int, default=128)
    parser.add_argument("--tower_dropout", type=float, default=0.0)
    parser.add_argument("--limit_rows", type=int, default=0, help="Smoke-test option: use only the first rows of each split file.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Opt in to mixed precision (bf16 on Ampere or newer, else fp16). The original trains in fp32.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if (out_dir / "final_results_test.csv").exists() and not args.overwrite:
        print(f"Completed run already exists, skipping: {out_dir}")
        return
    set_seed(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "run_config.json", vars(args))

    train_ds = AssayDataset(args.train_path, args.embeddings_dir, limit_rows=args.limit_rows)
    scaling = {"target_mean": train_ds.target_mean, "target_std": train_ds.target_std, "limit_rows": args.limit_rows}
    val_ds = AssayDataset(args.val_path, args.embeddings_dir, **scaling)
    test_ds = AssayDataset(args.test_path, args.embeddings_dir, **scaling)

    device = torch.device(args.device)
    amp_dtype = resolve_amp_dtype(device)[0] if args.amp else None
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    model = CachedMPEKRegressor(
        use_conditions=args.mode == "with_conditions",
        device=device,
        expert_dim=args.expert_dim,
        expert_layers=args.expert_layers,
        num_experts=args.num_experts,
        ple_layers=args.ple_layers,
        dropout=args.dropout,
        tower_layers=args.tower_layers,
        tower_hidden=args.tower_hidden,
        tower_dropout=args.tower_dropout,
    ).to(device)
    optimizer = ChildTuningAdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    inputs = features(train_ds, slice(None), device)
    target = train_ds.target.to(device)

    latest_path = out_dir / "checkpoint_last.pt"
    history_path = out_dir / "epoch_metrics.csv"
    start_epoch, history = 0, []
    if latest_path.exists() and not args.overwrite:
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        history = pd.read_csv(history_path).to_dict("records")
        print(f"Resuming from epoch {start_epoch}")

    run_label = f"{args.mode}/{args.split_name}/seed{args.seed}"
    started = time.time()
    epochs = progress(range(start_epoch + 1, args.epochs + 1), desc=f"{run_label} epochs", unit="epoch", dynamic_ncols=True, disable=args.no_progress)
    for epoch in epochs:
        train_pred = train_epoch(model, optimizer, inputs, target, args.batch_size, amp_dtype, scaler)
        train_metrics = regression_metrics(train_ds.raw_target, train_pred.cpu().squeeze(1) * train_ds.target_std + train_ds.target_mean)
        val_metrics = regression_metrics(val_ds.raw_target, predict(model, val_ds, device))
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        pd.DataFrame(history).to_csv(history_path, index=False)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args)}, latest_path)
        print(f"epoch={epoch} train_rmse={train_metrics['rmse']:.6f} val_rmse={val_metrics['rmse']:.6f}", flush=True)

    for partition, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        metrics, rows, n_pairs = assay_and_pair_metrics(dataset, predict(model, dataset, device))
        metrics.update(
            {
                "mode": args.mode,
                "split_group": args.split_group,
                "split_name": args.split_name,
                "seed": args.seed,
                "n": len(dataset),
                "n_pairs": n_pairs,
                "elapsed_seconds": time.time() - started,
            }
        )
        rows.to_csv(out_dir / f"predictions_{partition}.csv", index=False)
        pd.DataFrame([metrics]).to_csv(out_dir / f"final_results_{partition}.csv", index=False)


if __name__ == "__main__":
    main()
