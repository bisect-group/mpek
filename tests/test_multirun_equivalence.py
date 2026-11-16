"""The multi-run trainer (emulator_bench/multirun_train.py) against the single-run path of train_single_split.py.

Run from baselines/mpek (see emulator_bench/verify_equivalence.sh):
    CUDA_VISIBLE_DEVICES= python -m pytest tests/test_multirun_equivalence.py -m 'not gpu'   # CPU, small model
    python -m pytest tests/test_multirun_equivalence.py -m gpu -s                             # full model, Triton kernel, CUDA graphs, real split

The single-run reference is the plain batch-1 loop with the original ChildTuningAdamW (the multi-tensor version on CUDA).
Dropout masks are injected into both paths, so the only differences are float rounding in matmuls.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emulator_bench import common, multirun_train as mr  # noqa: E402  (common adds MTLKcatKM to sys.path)
from emulator_bench import train_single_split as train  # noqa: E402
from emulator_bench.fast_train import ForeachChildTuningAdamW  # noqa: E402
from emulator_bench.modeling import CachedMPEKRegressor  # noqa: E402
from MTLKcatKM.utils import ChildTuningAdamW  # noqa: E402

SMALL = dict(mr.HEAD, expert_dim=32, tower_hidden=16, protein_dim=64, ligand_dim=24)
FULL = dict(mr.HEAD)
DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


class Mask(torch.nn.Module):
    """Stands for the dropout layer of an expert: multiplies by the mask set before the step."""

    def forward(self, x):
        return x * self.mask


class Single:
    """One run of the single-run path with injected dropout masks."""

    def __init__(self, seed, cfg, device):
        torch.manual_seed(seed)
        self.model = CachedMPEKRegressor(**cfg)
        experts = self.model.multi_block.layers[0].experts
        self.masks = [Mask() for _ in range(2 * cfg["num_experts"])]
        for expert, mask in zip([*experts.share_expert, *experts.task_experts[0]], self.masks):
            expert.expert.mlp[1] = mask
        self.width = cfg["expert_dim"]
        self.model.to(device)
        optimizer = ForeachChildTuningAdamW if torch.device(device).type == "cuda" else ChildTuningAdamW
        self.optimizer = optimizer(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6)

    def step(self, protein, ligand, target, mask):
        for i, layer in enumerate(self.masks):
            layer.mask = mask[i * self.width : (i + 1) * self.width]
        pred = self.model(protein[None], ligand[None]).float()
        loss = (pred - target).pow(2).sum() / (1 + 1e-5)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return pred.detach().reshape(())

    @torch.no_grad()
    def eval_batch1(self, protein, ligand):
        """Eval-mode predictions one row at a time, as the single-run path does."""
        self.model.eval()
        for layer in self.masks:
            layer.mask = 1.0
        return torch.cat([self.model(protein[i : i + 1], ligand[i : i + 1]).reshape(-1) for i in range(len(protein))])

    def train_epoch(self, rows, order, masks):
        return torch.stack([self.step(rows[0][i], rows[1][i], rows[2][i], masks[j]) for j, i in enumerate(order.tolist())])


def random_rows(n, cfg, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    protein, ligand = torch.randn(n, cfg.get("protein_dim", 1024), generator=g), torch.randn(n, cfg.get("ligand_dim", 300), generator=g)
    return [t.to(device) for t in (protein, ligand, torch.randn(n, 1, generator=g))]


def random_masks(shape, p=0.3, seed=1):
    return ((torch.rand(*shape, generator=torch.Generator().manual_seed(seed)) >= p).float() / (1 - p))


def injected(masks, device):
    """mask_fn for run_epoch that hands out masks[step] of the runs in order, for every call."""
    state = {"at": 0}

    def mask_fn(count, k):
        out = masks[state["at"] : state["at"] + count, :k].to(device)
        state["at"] += count
        return out

    return mask_fn


def max_diffs(stack, singles, cfg):
    """Largest abs diff of weights and Adam state between the stack and the single runs, and the same relative to the largest value."""
    out = {"weights": (0.0, 0.0), "exp_avg": (0.0, 0.0), "exp_avg_sq": (0.0, 0.0)}
    for r, single in enumerate(singles):
        copy = CachedMPEKRegressor(**cfg)
        optimizer = ChildTuningAdamW(copy.parameters())
        stack.store_run(r, copy, optimizer, 0)
        for a, b in zip(copy.parameters(), single.model.parameters()):
            pairs = {"weights": (a, b)}
            if single.optimizer.state[b]:
                pairs.update({k: (optimizer.state[a][k], single.optimizer.state[b][k]) for k in ("exp_avg", "exp_avg_sq")})
            for k, (x, y) in pairs.items():
                x, y = x.detach().cpu(), y.detach().cpu()
                diff = (x - y).abs().max().item()
                out[k] = (max(out[k][0], diff), max(out[k][1], diff / max(y.abs().max().item(), 1e-30)))
    return out


@pytest.mark.parametrize("cfg", [SMALL])
def test_stack_roundtrip_keeps_weights_and_adam_state(cfg):
    single = Single(7, cfg, "cpu")
    rows, masks = random_rows(8, cfg, "cpu"), random_masks((8, 2 * cfg["num_experts"] * cfg["expert_dim"]))
    single.train_epoch(rows, torch.arange(8), masks)
    stack = mr.StackedMPEK([CachedMPEKRegressor(**cfg), CachedMPEKRegressor(**cfg)], "cpu")
    stack.load_run(1, single.model, single.optimizer)
    model = CachedMPEKRegressor(**cfg)
    optimizer = ChildTuningAdamW(model.parameters())
    stack.store_run(1, model, optimizer, 8)
    for (name, a), b in zip(model.named_parameters(), single.model.parameters()):
        assert torch.equal(a, b), name
        if single.optimizer.state[b]:
            assert optimizer.state[a]["step"] == single.optimizer.state[b]["step"] == 8
            assert all(torch.equal(optimizer.state[a][k], single.optimizer.state[b][k]) for k in ("exp_avg", "exp_avg_sq"))
        else:
            assert not optimizer.state[a]  # the share gate feeds no output, so it has no gradient and no state


def test_initial_weights_are_the_single_run_weights():
    """Weights from set_seed(seed) and the model constructor are what the stack holds, for every seed."""
    models = []
    for seed in (666, 777, 888):
        common.set_seed(seed)
        models.append(CachedMPEKRegressor(**SMALL))
    stack = mr.StackedMPEK(models, "cpu")
    for r, seed in enumerate((666, 777, 888)):
        common.set_seed(seed)
        fresh = CachedMPEKRegressor(**SMALL)
        copy = CachedMPEKRegressor(**SMALL)
        stack.store_run(r, copy, ChildTuningAdamW(copy.parameters()), 0)
        assert all(torch.equal(a, b) for a, b in zip(fresh.parameters(), copy.parameters()))


def test_epoch_orders_are_the_single_run_orders():
    """Orders drawn per run from saved CPU random states equal the script's draws, also when the runs are interleaved."""
    sizes, epochs, seeds = {666: 37, 777: 52}, 4, (666, 777)
    expected = {}
    for seed in seeds:
        common.set_seed(seed)
        CachedMPEKRegressor(**SMALL)
        expected[seed] = []
        for _ in range(epochs):
            expected[seed].append(train.epoch_order(sizes[seed]))
            train.draw_loader_seed()
            train.draw_loader_seed()
        expected[seed].append(torch.rand(()))  # the stream continues the same way after the epochs
    states = {}
    for seed in seeds:
        common.set_seed(seed)
        CachedMPEKRegressor(**SMALL)
        states[seed] = torch.get_rng_state()
    for epoch in range(epochs):
        for seed in seeds:
            order, states[seed] = mr.draw_epoch(states[seed], sizes[seed])
            assert torch.equal(order, expected[seed][epoch])
    for seed in seeds:
        torch.set_rng_state(states[seed])
        assert torch.equal(torch.rand(()), expected[seed][epochs])


def test_gradients_equal_autograd():
    """The hand-written backward gives the autograd gradient of every parameter, run by run."""
    cfg, k = SMALL, 3
    singles = [Single(seed, cfg, "cpu") for seed in range(k)]
    stack = mr.StackedMPEK([s.model for s in singles], "cpu")
    rows = random_rows(k, cfg, "cpu")
    masks = random_masks((k, 2 * cfg["num_experts"] * cfg["expert_dim"]))
    pred, grads = stack.forward_backward(rows[0], rows[1], masks, rows[2])
    for r, single in enumerate(singles):
        single.optimizer.zero_grad(set_to_none=True)
        for i, layer in enumerate(single.masks):
            layer.mask = masks[r, i * cfg["expert_dim"] : (i + 1) * cfg["expert_dim"]]
        y = single.model(rows[0][r : r + 1], rows[1][r : r + 1])
        ((y - rows[2][r : r + 1]).pow(2).sum() / (1 + 1e-5)).backward()
        assert torch.allclose(pred[r], y.detach().reshape(()), rtol=1e-5, atol=1e-6)
        worst = 0.0
        for key, (d, x) in grads.items():
            slots = [(param, (d[r, slot, None] * x[r][None]).reshape(param.shape)) for param, slot in stack._slots(single.model, key)]
            scale = max(param.grad.abs().max().item() for param, _ in slots)  # per tensor: one expert can have a nearly zero gradient
            worst = max(worst, max((got - param.grad).abs().max().item() for param, got in slots) / scale)
        assert worst < 1e-5, worst


@pytest.mark.parametrize("device, cfg", [("cpu", SMALL), pytest.param("cuda:0", FULL, marks=pytest.mark.gpu)])
def test_training_equals_single_run(device, cfg):
    """3 runs, 5 epochs of 400 rows (2,000 steps), identical init, orders and dropout masks, against the single-run loop."""
    seeds, n, epochs = (666, 777, 888), 400, 5
    singles = [Single(seed, cfg, device) for seed in seeds]
    models = []
    for seed in seeds:
        torch.manual_seed(seed)
        models.append(CachedMPEKRegressor(**cfg))
    rows = random_rows(n, cfg, device)
    stack = mr.StackedMPEK(models, device, data=tuple(rows), max_rows=n)
    width = 2 * cfg["num_experts"] * cfg["expert_dim"]
    masks = random_masks((epochs * n, len(seeds), width))
    mask_fn, generator = injected(masks, device), torch.Generator().manual_seed(3)
    worst_pred = {}
    for epoch in range(epochs):
        orders = torch.stack([torch.randperm(n, generator=generator) for _ in seeds], 1)
        pred = stack.run_epoch(orders.to(device), [n] * len(seeds), epoch, mask_fn)
        for r, single in enumerate(singles):
            expected = single.train_epoch(rows, orders[:, r], masks[epoch * n : (epoch + 1) * n, r].to(device))
            diff = (pred[:, r] - expected).abs()
            worst_pred[(epoch, r)] = (diff.max().item(), (diff / expected.abs().clamp_min(1.0)).max().item())
        if epoch == 2:
            mid = max_diffs(stack, singles, cfg)
    final = max_diffs(stack, singles, cfg)
    val = random_rows(100, cfg, device, seed=9)
    pred_diff = max((stack.predict(r, val[0], val[1]).reshape(-1) - single.eval_batch1(val[0], val[1])).abs().max().item() for r, single in enumerate(singles))
    print(f"\n{device}: {epochs * n} steps, 3 runs. After 1,200 steps: weights abs {mid['weights'][0]:.1e} rel {mid['weights'][1]:.1e}, exp_avg rel {mid['exp_avg'][1]:.1e}")
    print(f"after {epochs * n} steps: weights abs {final['weights'][0]:.1e} rel {final['weights'][1]:.1e}, exp_avg rel {final['exp_avg'][1]:.1e}, exp_avg_sq rel {final['exp_avg_sq'][1]:.1e}")
    print(f"train predictions: max abs {max(a for a, _ in worst_pred.values()):.1e}, max rel {max(r for _, r in worst_pred.values()):.1e}; eval predictions after training: max abs {pred_diff:.1e}")
    assert mid["weights"][1] < 1e-5 and max(r for (e, _), (_, r) in worst_pred.items() if e <= 2) < 1e-5
    if device == "cpu":  # the full model on random targets amplifies rounding differences after about 1,500 steps, as a 1e-7 weight change does
        assert final["weights"][1] < 1e-5 and max(r for _, r in worst_pred.values()) < 1e-5 and pred_diff < 1e-5


def test_finished_runs_take_no_update():
    """A run outside the active prefix keeps weights, Adam state and nothing else changes, bit for bit."""
    cfg, R = SMALL, 3
    models = [CachedMPEKRegressor(**cfg) for _ in range(R)]
    rows = random_rows(8, cfg, "cpu")
    stack = mr.StackedMPEK(models, "cpu")
    width = 2 * cfg["num_experts"] * cfg["expert_dim"]
    masks = random_masks((6, R, width))
    for k in (3, 3, 2, 2, 1, 1):  # the sizes of an epoch with sizes (6, 4, 2)
        before = {name: {key: t.clone() for key, t in table.items()} for name, table in (("p", stack.p), ("m", stack.m), ("v", stack.v))}
        stack.train_step(rows[0][:k], rows[1][:k], masks[0, :k], rows[2][:k], stack.constant_step[:k])
        for name, table in (("p", stack.p), ("m", stack.m), ("v", stack.v)):
            for key in table:
                assert torch.equal(table[key][k:], before[name][key][k:]), (name, key, k)
                if k == 3 and key in mr.TRAINABLE:
                    assert not torch.equal(table[key][:k], before[name][key][:k])


def test_uneven_sizes_equal_single_runs_of_their_own_length():
    """Runs of 60, 45 and 30 rows over 3 epochs: each run takes exactly size * epochs steps, with the step sizes of its own count."""
    cfg, sizes, epochs = SMALL, [60, 45, 30], 3
    seeds = (666, 777, 888)
    singles = [Single(seed, cfg, "cpu") for seed in seeds]
    models = []
    for seed in seeds:
        torch.manual_seed(seed)
        models.append(CachedMPEKRegressor(**cfg))
    rows = random_rows(60, cfg, "cpu")
    stack = mr.StackedMPEK(models, "cpu", data=tuple(rows), max_rows=60)
    width = 2 * cfg["num_experts"] * cfg["expert_dim"]
    masks = random_masks((epochs * 60, 3, width))
    generator = torch.Generator().manual_seed(4)
    at = {"epoch": 0}

    def mask_fn(count, k):  # step j of epoch e hands run r the mask [e * 60 + j, r]
        out = masks[at["epoch"] * 60 + at["j"] : at["epoch"] * 60 + at["j"] + count, :k]
        at["j"] += count
        return out

    for epoch in range(epochs):
        orders = [torch.randperm(n, generator=generator) for n in sizes]
        at.update(epoch=epoch, j=0)
        stack.run_epoch(torch.stack([torch.nn.functional.pad(o, (0, 60 - len(o))) for o in orders], 1), sizes, epoch, mask_fn)
        for r, single in enumerate(singles):
            single.train_epoch(rows, orders[r], masks[epoch * 60 : epoch * 60 + sizes[r], r])
    diffs = max_diffs(stack, singles, cfg)
    print(f"\nuneven sizes {sizes}: weights rel {diffs['weights'][1]:.1e}, exp_avg rel {diffs['exp_avg'][1]:.1e}, exp_avg_sq rel {diffs['exp_avg_sq'][1]:.1e}")
    assert diffs["weights"][1] < 1e-5 and diffs["exp_avg_sq"][1] < 1e-5


def tiny_split(tmp_path, rng):
    """Split files and an embedding cache of random vectors in the real cache layout (dims of SMALL)."""
    rows = []
    for name, n in (("train", 40), ("val", 12), ("test", 10)):
        frame = pd.DataFrame({"sequence": [f"MKT{i}{name}AY" for i in range(n)], "smiles": [f"C{'C' * (i % 5)}O{name}" for i in range(n)], "log10_value": rng.normal(size=n)})
        frame.to_parquet(tmp_path / f"{name}.parquet")
        rows.append(frame)
    frame = pd.concat(rows)
    for sequence in frame["sequence"].unique():
        common.save_npy_atomic(common.protein_cache_path(tmp_path / "emb", common.protein_text(sequence)), rng.normal(size=SMALL["protein_dim"]))
    for smiles in frame["smiles"].unique():
        common.save_npy_atomic(common.ligand_cache_path(tmp_path / "emb", smiles), rng.normal(size=SMALL["ligand_dim"]))
    return {"task": "kcat", "split_group": "g", "split_name": "all", **{f"{n}_path": str(tmp_path / f"{n}.parquet") for n in ("train", "val", "test")}}


def test_group_files_and_resume(tmp_path, monkeypatch):
    """The group writes the single-run files and a stopped group resumes to the same result as an uninterrupted one."""
    monkeypatch.setattr(mr, "HEAD", SMALL)
    job = tiny_split(tmp_path, np.random.default_rng(0))

    def runs(root):
        return [{"job": job, "seed": seed, "out_dir": tmp_path / root / f"seed_{seed}", "overwrite": False} for seed in (1, 2, 3)]

    mr.train_group(runs("whole"), str(tmp_path / "emb"), "cpu", epochs=3)
    mr.train_group(runs("part"), str(tmp_path / "emb"), "cpu", epochs=2)
    for run in runs("part"):
        (run["out_dir"] / "final_results_test.csv").unlink()
    mr.train_group(runs("part"), str(tmp_path / "emb"), "cpu", epochs=3)
    files = {"checkpoint_last.pt", "epoch_metrics.csv", "run_config.json", "run_settings.json"} | {f"{kind}_{name}.csv" for kind in ("final_results", "predictions") for name in ("train", "val", "test")}
    for seed in (1, 2, 3):
        whole, part = (tmp_path / root / f"seed_{seed}" for root in ("whole", "part"))
        assert {p.name for p in whole.iterdir()} == files
        assert len(pd.read_csv(part / "epoch_metrics.csv")) == 3 and pd.read_csv(whole / "final_results_test.csv")["seed"][0] == seed
        np.testing.assert_allclose(pd.read_csv(part / "predictions_test.csv"), pd.read_csv(whole / "predictions_test.csv"), rtol=1e-5)
        checkpoint = torch.load(whole / "checkpoint_last.pt", map_location="cpu")
        assert checkpoint["epoch"] == 3 and len(checkpoint["optimizer_state_dict"]["state"]) == 24  # all but the 2 share-gate tensors
    seeds = [pd.read_csv(tmp_path / "whole" / f"seed_{s}" / "predictions_test.csv")["y_pred"] for s in (1, 2, 3)]
    assert not np.allclose(seeds[0], seeds[1])


@pytest.mark.gpu
def test_adam_kernel_equals_the_optimizers_bit_for_bit():
    """The Triton update on random rank-one gradients against ForeachChildTuningAdamW and the original ChildTuningAdamW, 30 steps."""
    device, R = "cuda:0", 2
    stack = mr.StackedMPEK([CachedMPEKRegressor(**FULL) for _ in range(R)], device)
    refs = {cls: [CachedMPEKRegressor(**FULL).to(device) for _ in range(R)] for cls in (ForeachChildTuningAdamW, ChildTuningAdamW)}
    for cls, models in refs.items():
        for r, model in enumerate(models):
            template = CachedMPEKRegressor(**FULL)
            stack.store_run(r, template, ChildTuningAdamW(template.parameters()), 0)
            model.load_state_dict(template.state_dict())
    optimizers = {cls: [cls(m.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6) for m in models] for cls, models in refs.items()}
    g = torch.Generator(device=device).manual_seed(0)
    for t in range(1, 31):
        grads = {}
        for key in mr.TRAINABLE:
            n_out, n_in = stack.p[key].shape[1:]
            d = torch.randn(R, n_out, device=device, generator=g) * (torch.rand(R, n_out, device=device, generator=g) > 0.5)
            grads[key] = (d, torch.randn(R, n_in, device=device, generator=g) if n_in > 1 else torch.ones(R, 1, device=device))
        step = stack.step_sizes(torch.full((R,), t, device=device))
        for key, (d, x) in grads.items():
            stack.adam(key, R, d, x, step)
        for cls, models in refs.items():
            for r, model in enumerate(models):
                for key, (d, x) in grads.items():
                    for param, rows in stack._slots(model, key):
                        param.grad = (d[r, rows, None] * x[r][None]).reshape(param.shape)
                optimizers[cls][r].step()
    for cls, models in refs.items():
        for r, model in enumerate(models):
            copy = CachedMPEKRegressor(**FULL)
            optimizer = ChildTuningAdamW(copy.parameters())
            stack.store_run(r, copy, optimizer, 30)
            for (name, a), b in zip(copy.named_parameters(), model.parameters()):
                assert torch.equal(a.to(device), b), (cls.__name__, name)
                if optimizers[cls][r].state[b]:
                    assert all(torch.equal(optimizer.state[a][k].to(device), optimizers[cls][r].state[b][k]) for k in ("exp_avg", "exp_avg_sq"))
    print("\nTriton update vs ForeachChildTuningAdamW and ChildTuningAdamW: 30 steps, weights and Adam state identical")


@pytest.mark.gpu
def test_cuda_graph_epoch_equals_eager_epoch():
    """16,000 steps: eager steps up to the constant step size, then 50-step graphs. The state equals the eager loop bit for bit."""
    device, sizes, R = "cuda:0", [8000, 7990], 2  # the last 10 steps have one active run
    rows = random_rows(8000, SMALL, device)
    width = 2 * SMALL["num_experts"] * SMALL["expert_dim"]
    masks = random_masks((8000, R, width))
    results = []
    for use_graph in (False, True):
        models = []
        for seed in (666, 777):
            torch.manual_seed(seed)
            models.append(CachedMPEKRegressor(**SMALL))
        stack = mr.StackedMPEK(models, device, data=tuple(rows), max_rows=8000, use_graph=use_graph)
        order = torch.stack([torch.randperm(8000, generator=torch.Generator().manual_seed(s)) for s in (1, 2)], 1).to(device)
        pred = stack.run_epoch(order, sizes, 1, injected(masks, device)).clone()  # step counts 8,001 to 16,000
        results.append((pred, {n: {k: t.clone() for k, t in table.items()} for n, table in (("p", stack.p), ("m", stack.m), ("v", stack.v))}))
    assert torch.equal(results[0][0], results[1][0])
    assert all(torch.equal(results[0][1][n][k], results[1][1][n][k]) for n in "pmv" for k in results[0][1][n])
    print("\nCUDA graphs vs eager steps over 8,000 steps of 2 runs of different length (switch at step 15,901): predictions, weights and Adam state identical")


@pytest.mark.gpu
def test_downstream_validation_metrics_on_a_real_split(tmp_path):
    """Real kcat split, 3 seeds, 3 epochs with normal dropout: per-epoch validation RMSE of the group against single runs."""
    job = common.discover_split_jobs(common.DEFAULT_BASE_DIR, tasks=["kcat"], split_groups=["random_splits_grouped_sequence"])[0]
    seeds, epochs = (666, 777, 888), 3
    mr.train_group([{"job": job, "seed": s, "out_dir": tmp_path / "multi" / f"seed_{s}", "overwrite": False} for s in seeds], str(common.DEFAULT_EMBEDDINGS_DIR), "cuda:0", epochs=epochs)
    for s in seeds:
        subprocess.run(
            [sys.executable, str(ROOT / "emulator_bench" / "train_single_split.py"), "--train_path", job["train_path"], "--val_path", job["val_path"], "--test_path", job["test_path"],
             "--embeddings_dir", str(common.DEFAULT_EMBEDDINGS_DIR), "--out_dir", str(tmp_path / "single" / f"seed_{s}"), "--seed", str(s), "--epochs", str(epochs), "--no_progress"],
            check=True, cwd=str(ROOT),
        )  # fmt: skip
    val = {kind: np.array([pd.read_csv(tmp_path / kind / f"seed_{s}" / "epoch_metrics.csv")["val_rmse"] for s in seeds]) for kind in ("multi", "single")}
    for epoch in range(epochs):
        gap = abs(val["multi"][:, epoch].mean() - val["single"][:, epoch].mean())
        spread = val["single"][:, epoch].std(ddof=1)
        print(f"\nepoch {epoch + 1}: val RMSE multi {val['multi'][:, epoch].round(4)} single {val['single'][:, epoch].round(4)}; gap of means {gap:.4f}, seed spread {spread:.4f}")
        assert gap <= 2 * spread
    for kind in ("multi", "single"):
        assert (tmp_path / kind / "seed_666" / "predictions_test.csv").exists()
