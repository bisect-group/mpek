import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from emulator_bench.common import regression_metrics, resolve_amp_dtype, save_json, set_seed
from emulator_bench.dataset import CachedMPEKDataset
from emulator_bench.modeling import CachedMPEKRegressor
from MTLKcatKM.utils import ChildTuningAdamW
from src.utils.rich_progress import progress


def draw_loader_seed() -> None:
    """Advance the global RNG by one draw, as each pass of an original torch DataLoader does."""
    torch.empty((), dtype=torch.int64).random_()


@torch.no_grad()
def predict(model, dataset, device, batch_size):
    """Sequential pass in the training batch size. PLE outputs depend on the batch size, as in the original."""
    model.eval()
    preds = [
        model(dataset.protein[start : start + batch_size].to(device), dataset.ligand[start : start + batch_size].to(device)).cpu()
        for start in range(0, len(dataset), batch_size)
    ]
    return torch.cat(preds).squeeze(1) * dataset.target_std + dataset.target_mean


def train_epoch(model, optimizer, data, batch_size, amp_dtype=None, scaler=None):
    """One shuffled pass over device-resident tensors; returns the dropout-mode predictions.

    The global RNG draws match the original RandomSampler loader: one iterator seed, one sampler seed,
    then a permutation from a generator seeded with the sampler seed.
    """
    protein, ligand, target = data
    model.train()
    draw_loader_seed()
    generator = torch.Generator().manual_seed(int(torch.empty((), dtype=torch.int64).random_()))
    order = torch.randperm(len(target), generator=generator).to(target.device)
    train_pred = torch.zeros_like(target)
    for start in range(0, len(target) - batch_size + 1, batch_size):
        idx = order[start : start + batch_size]
        with torch.autocast(device_type=target.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(protein[idx], ligand[idx])
        pred = pred.float()
        loss = (pred - target[idx]).pow(2).sum() / (len(idx) + 1e-5)  # original compute_loss form
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


def main():
    parser = argparse.ArgumentParser(description="Train one cached MPEK split with the original MPEK settings.")
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
    settings = {"epochs": args.epochs, "limit_rows": args.limit_rows}
    settings_path = out_dir / "run_settings.json"
    if (out_dir / "final_results_test.csv").exists() and not args.overwrite:
        if settings_path.exists() and json.loads(settings_path.read_text()) == settings:
            print(f"Completed run already exists, skipping: {out_dir}")
            return
        print(f"Completed run used other settings, rerunning: {out_dir}")
        args.overwrite = True
    set_seed(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "run_config.json", vars(args))

    common = (args.embeddings_dir, args.sequence_col, args.smiles_col, args.target_col)
    train_ds = CachedMPEKDataset(args.train_path, *common, limit_rows=args.limit_rows)
    scaling = {"target_mean": train_ds.target_mean, "target_std": train_ds.target_std, "limit_rows": args.limit_rows}
    val_ds = CachedMPEKDataset(args.val_path, *common, **scaling)
    test_ds = CachedMPEKDataset(args.test_path, *common, **scaling)

    device = torch.device(args.device)
    amp_dtype = resolve_amp_dtype(device)[0] if args.amp else None
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
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
    optimizer = ChildTuningAdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    data = (train_ds.protein.to(device), train_ds.ligand.to(device), train_ds.target.to(device))

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

    run_label = f"{args.task}/{args.split_name}/seed{args.seed}"
    started = time.time()
    epochs = progress(range(start_epoch + 1, args.epochs + 1), desc=f"{run_label} epochs", unit="epoch", dynamic_ncols=True, disable=args.no_progress)
    for epoch in epochs:
        train_pred = train_epoch(model, optimizer, data, args.batch_size, amp_dtype, scaler)
        train_metrics = regression_metrics(train_ds.raw_target, (train_pred.cpu().squeeze(1) * train_ds.target_std + train_ds.target_mean))
        draw_loader_seed()
        val_metrics = regression_metrics(val_ds.raw_target, predict(model, val_ds, device, args.batch_size))
        draw_loader_seed()  # the original also evaluates the test loader each epoch
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        pd.DataFrame(history).to_csv(history_path, index=False)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args)}, latest_path)
        print(f"epoch={epoch} train_rmse={train_metrics['rmse']:.6f} val_rmse={val_metrics['rmse']:.6f}", flush=True)

    for split_name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        y_pred = predict(model, dataset, device, args.batch_size)
        metrics = regression_metrics(dataset.raw_target, y_pred)
        metrics.update(
            {
                "task": args.task,
                "split_group": args.split_group,
                "split_name": args.split_name,
                "seed": args.seed,
                "n": len(dataset),
                "rows_dropped": dataset.rows_dropped,
                "elapsed_seconds": time.time() - started,
            }
        )
        pd.DataFrame([metrics]).to_csv(out_dir / f"final_results_{split_name}.csv", index=False)
        pd.DataFrame({"y_true": dataset.raw_target.numpy(), "y_pred": y_pred.numpy()}).to_csv(out_dir / f"predictions_{split_name}.csv", index=False)

    save_json(settings_path, settings)


if __name__ == "__main__":
    main()
