"""Train many independent batch-1 MPEK runs together on one device.

Every run keeps the batch-1 maths of `train_single_split.py`: its own initial weights, its own shuffled sample order, its own
dropout masks, MSE loss on its own normalised target, and a ChildTuningAdamW step per sample. The runs only share kernels.
Parameters and Adam state of the R runs are stacked along a leading run axis. Batched matmuls do the forward and the
hand-written backward. One kernel updates a parameter of all active runs from the rank-one gradient `d x x^T`.

Runs are sorted by train size, largest first. At step j the active runs are the first k runs, a prefix. Runs that finished
their epoch are not touched: their weights, Adam state and step count stay as they are.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import triton
import triton.language as tl

from emulator_bench.common import regression_metrics, save_json, set_seed
from emulator_bench.dataset import CachedMPEKDataset
from emulator_bench.fast_train import GRAPH_STEPS, _graph, _warm_up, constant_step_after
from emulator_bench.modeling import CachedMPEKRegressor
from emulator_bench.train_single_split import draw_loader_seed, epoch_order
from MTLKcatKM.utils import ChildTuningAdamW

LOSS_DIVISOR = 1 + 1e-5  # the original compute_loss divides the summed squared error by (batch size + 1e-5)
EXPERT = "multi_block.layers.0.experts.{}.expert.mlp.0.{}"
TRAINABLE = ("proj_w", "proj_b", "w1", "b1", "gate_w", "gate_b", "tower1_w", "tower1_b", "tower2_w", "tower2_b")


def parameter_names(num_experts: int) -> dict[str, list[str]]:
    """Model parameters that form each stacked tensor, in row order. The share gate feeds no output: it never gets a gradient."""
    experts = [f"share_expert.{i}" for i in range(num_experts)] + [f"task_experts.0.{i}" for i in range(num_experts)]
    gate = "multi_block.layers.0.gates.{}.fc1.{}"
    return {
        "proj_w": ["protein_projection.weight"], "proj_b": ["protein_projection.bias"],
        "w1": [EXPERT.format(e, "weight") for e in experts], "b1": [EXPERT.format(e, "bias") for e in experts],
        "gate_w": [gate.format("task_gates.0", "weight")], "gate_b": [gate.format("task_gates.0", "bias")],
        "tower1_w": ["tower.mlp.0.weight"], "tower1_b": ["tower.mlp.0.bias"],
        "tower2_w": ["tower.mlp.3.weight"], "tower2_b": ["tower.mlp.3.bias"],
        "share_gate_w": [gate.format("share_gate", "weight")], "share_gate_b": [gate.format("share_gate", "bias")],
    }  # fmt: skip


@triton.jit
def _adam_rank_one(p_ptr, m_ptr, v_ptr, d_ptr, x_ptr, step_ptr, n_out, n_in, b1, ob1, b2, ob2, eps, decay, BLOCK: tl.constexpr):
    """ChildTuningAdamW on p[run, o, i] with gradient d[run, o] * x[run, i]. The op order follows the multi-tensor kernels."""
    run = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_out * n_in
    row = offs // n_in
    g = tl.load(d_ptr + run * n_out + row, mask=valid) * tl.load(x_ptr + run * n_in + offs - row * n_in, mask=valid)
    at = run.to(tl.int64) * n_out * n_in + offs
    m = tl.fma(g, ob1, tl.load(m_ptr + at, mask=valid) * b1)
    v = tl.fma(g * g, ob2, tl.load(v_ptr + at, mask=valid) * b2)
    p = tl.fma(-tl.load(step_ptr + run), tl.div_rn(m, tl.sqrt_rn(v) + eps), tl.load(p_ptr + at, mask=valid))
    tl.store(m_ptr + at, m, mask=valid)
    tl.store(v_ptr + at, v, mask=valid)
    tl.store(p_ptr + at, tl.fma(p, decay, p), mask=valid)


class StackedMPEK:
    """Parameters and Adam state of R runs. Biases are stored as [R, O, 1], weights as [R, O, I]."""

    def __init__(self, models: list[CachedMPEKRegressor], device, lr=1e-4, betas=(0.9, 0.999), eps=1e-6, weight_decay=1e-6, dropout=0.3, data=None, max_rows=0, use_graph=False):
        """`data` = (protein, ligand, target) tensors on the device, the rows `run_epoch` indexes. `max_rows` = the longest train set."""
        self.R, self.device, self.dropout = len(models), torch.device(device), dropout
        self.data, self.use_graph, self.graphs = data, use_graph, None
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.num_experts = len(models[0].multi_block.layers[0].experts.share_expert)
        self.names = parameter_names(self.num_experts)
        self.p = {key: torch.stack([self._gather(m, names) for m in models]).to(self.device) for key, names in self.names.items()}
        self.m = {key: torch.zeros_like(self.p[key]) for key in TRAINABLE}
        self.v = {key: torch.zeros_like(self.p[key]) for key in TRAINABLE}
        self.ones = torch.ones(self.R, 1, device=self.device)
        self.last_constant = constant_step_after(lr, *betas)
        t = torch.arange(1, self.last_constant + 2, dtype=torch.float64)
        # step size of every step in fp32, as the scalar the multi-tensor optimizer passes; the last entry is the constant one
        self.step_table = (lr * (1.0 - betas[1] ** t).sqrt() / (1.0 - betas[0] ** t)).float().to(self.device)
        self.constant_step = self.step_table[-1].expand(self.R).contiguous()
        self.use_triton = self.device.type == "cuda"
        self.generator = torch.Generator(self.device)
        self.order = torch.zeros(max_rows, self.R, dtype=torch.long, device=self.device)
        self.pred = torch.zeros(max_rows, self.R, device=self.device)
        self.ptr, self.mask_ptr = torch.zeros(1, dtype=torch.long, device=self.device), torch.zeros(1, dtype=torch.long, device=self.device)
        self.mask_buf = torch.zeros(GRAPH_STEPS, self.R, self.p["b1"].shape[1], device=self.device)

    @staticmethod
    def _gather(model, names):
        params = dict(model.named_parameters())
        return torch.cat([params[n].detach().reshape(params[n].shape[0], -1) for n in names])

    def _slots(self, model, key):
        """(parameter, rows of the stacked tensor) for each parameter of a stacked tensor."""
        params, row = dict(model.named_parameters()), 0
        for name in self.names[key]:
            yield params[name], slice(row, row + params[name].shape[0])
            row += params[name].shape[0]

    @torch.no_grad()
    def load_run(self, r, model, optimizer):
        """Copy weights, and the Adam state if the optimizer has it, of one run into the stack."""
        for key in self.names:
            for param, rows in self._slots(model, key):
                self.p[key][r, rows] = param.reshape(-1, self.p[key].shape[-1]).to(self.device)
                if key in self.m and optimizer.state[param]:
                    for table, name in ((self.m, "exp_avg"), (self.v, "exp_avg_sq")):
                        table[key][r, rows] = optimizer.state[param][name].reshape(-1, self.p[key].shape[-1]).to(self.device)

    @torch.no_grad()
    def store_run(self, r, model, optimizer, steps):
        """Copy one run out of the stack into a model and a ChildTuningAdamW, as the single-run checkpoint holds them."""
        for key in self.names:
            for param, rows in self._slots(model, key):
                param.copy_(self.p[key][r, rows].reshape(param.shape))
                if key in self.m:
                    optimizer.state[param] = {
                        "step": steps,
                        "exp_avg": self.m[key][r, rows].reshape(param.shape).clone(),
                        "exp_avg_sq": self.v[key][r, rows].reshape(param.shape).clone(),
                    }

    def random_masks(self, count, k):
        """Dropout masks (0 or 1/(1-p)) for `count` steps of k runs: [count, k, 2 * num_experts * expert_dim]."""
        keep = torch.rand(count, k, self.p["b1"].shape[1], device=self.device, generator=self.generator)
        return (keep >= self.dropout).float() / (1.0 - self.dropout)

    def forward_backward(self, prot, lig, mask, target):
        """One batch-1 step of k runs (the first k): prediction [k] and the rank-one gradient factors (d, x) per parameter."""
        k = len(prot)
        w = {key: t[:k] for key, t in self.p.items()}
        ne = 2 * self.num_experts
        xp = torch.baddbmm(w["proj_b"], w["proj_w"], prot[:, :, None])
        x = torch.cat([lig[:, :, None], xp], 1)
        hidden = torch.baddbmm(w["b1"], w["w1"], x) * mask[:, :, None]
        experts = hidden.clamp_min(0).view(k, ne, -1)
        gate = torch.baddbmm(w["gate_b"], w["gate_w"], x)
        mixed = (gate * experts).sum(1, keepdim=True).transpose(1, 2)
        pre = torch.baddbmm(w["tower1_b"], w["tower1_w"], mixed)
        act = pre.clamp_min(0)
        y = torch.baddbmm(w["tower2_b"], w["tower2_w"], act)
        dy = (y - target[:, :, None]) * (2.0 / LOSS_DIVISOR)
        d_act = torch.bmm(w["tower2_w"].transpose(1, 2), dy) * (pre > 0)
        d_mixed = torch.bmm(w["tower1_w"].transpose(1, 2), d_act)
        d_gate = (experts * d_mixed.transpose(1, 2)).sum(-1, keepdim=True)
        d_hidden = (gate * d_mixed.transpose(1, 2)).reshape(k, -1, 1) * (hidden > 0) * mask[:, :, None]
        d_x = torch.baddbmm(torch.bmm(w["gate_w"].transpose(1, 2), d_gate), w["w1"].transpose(1, 2), d_hidden)
        d_proj = d_x[:, lig.shape[1] :]
        flat = lambda t: t.reshape(k, -1).contiguous()  # noqa: E731
        ones = self.ones[:k]
        grads = {
            "proj_w": (flat(d_proj), prot), "proj_b": (flat(d_proj), ones), "w1": (flat(d_hidden), flat(x)), "b1": (flat(d_hidden), ones),
            "gate_w": (flat(d_gate), flat(x)), "gate_b": (flat(d_gate), ones), "tower1_w": (flat(d_act), flat(mixed)), "tower1_b": (flat(d_act), ones),
            "tower2_w": (flat(dy), flat(act)), "tower2_b": (flat(dy), ones),
        }  # fmt: skip
        return y.reshape(k), grads

    @torch.no_grad()
    def adam(self, key, k, d, x, step):
        """ChildTuningAdamW (no bias-correction shortcuts) on parameter `key` of the first k runs; step[r] is the fp32 step size."""
        p, m, v = self.p[key][:k], self.m[key][:k], self.v[key][:k]
        (b1, b2), decay = self.betas, -self.lr * self.weight_decay
        if self.use_triton:
            n_out, n_in = p.shape[1:]
            grid = (triton.cdiv(n_out * n_in, 1024), k)
            _adam_rank_one[grid](p, m, v, d, x, step, n_out, n_in, b1, 1.0 - b1, b2, 1.0 - b2, self.eps, decay, BLOCK=1024, enable_fp_fusion=False)
            return
        g = d[:, :, None] * x[:, None, :]
        m.mul_(b1).add_(g, alpha=1.0 - b1)
        v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
        p.sub_(m / (v.sqrt() + self.eps) * step.view(-1, 1, 1))
        p.add_(p, alpha=decay)

    def step_sizes(self, t):
        """fp32 step sizes for the 1-based optimizer step counts t (a tensor, one per run)."""
        return self.step_table[(t - 1).clamp(max=self.last_constant)]

    def train_step(self, prot, lig, mask, target, step):
        """Forward, backward and update of the first len(prot) runs on one row each. Returns the predictions (dropout mode)."""
        y, grads = self.forward_backward(prot, lig, mask, target)
        for key in TRAINABLE:
            self.adam(key, len(prot), *grads[key], step)
        return y

    @torch.no_grad()
    def predict(self, r, prot, lig, chunk=4096):
        """Eval-mode predictions [N, 1] of run r. Each row is independent, so this equals a batch-1 pass up to rounding."""
        w = {key: t[r] for key, t in self.p.items()}
        out = []
        for s in range(0, len(prot), chunk):
            x = torch.cat([lig[s : s + chunk], torch.addmm(w["proj_b"][:, 0], prot[s : s + chunk], w["proj_w"].T)], 1)
            experts = torch.addmm(w["b1"][:, 0], x, w["w1"].T).clamp_min(0).view(len(x), 2 * self.num_experts, -1)
            gate = torch.addmm(w["gate_b"][:, 0], x, w["gate_w"].T)
            mixed = (gate[:, :, None] * experts).sum(1)
            act = torch.addmm(w["tower1_b"][:, 0], mixed, w["tower1_w"].T).clamp_min(0)
            out.append(torch.addmm(w["tower2_b"][:, 0], act, w["tower2_w"].T))
        return torch.cat(out)

    def _graph_body(self):
        prot, lig, target = self.data
        idx = self.order.index_select(0, self.ptr)[0]
        mask = self.mask_buf.index_select(0, self.mask_ptr)[0]
        y = self.train_step(prot.index_select(0, idx), lig.index_select(0, idx), mask, target.index_select(0, idx), self.constant_step)
        self.pred.index_copy_(0, self.ptr, y[None])
        self.ptr += 1
        self.mask_ptr += 1

    def _capture(self):
        """Capture 50-step and 1-step graphs of all R runs. Weights and Adam state are restored afterwards."""
        tensors = [self.pred] + [t for table in (self.p, self.m, self.v) for key, t in table.items() if key in TRAINABLE]
        saved = [t.clone() for t in tensors]
        restore = lambda: [t.copy_(s) for t, s in zip(tensors, saved)]  # noqa: E731
        _warm_up(self._graph_body)
        restore()
        self.graphs = {count: _graph(self._graph_body, count) for count in (GRAPH_STEPS, 1)}
        restore()

    def run_epoch(self, order, sizes, epoch, mask_fn=None):
        """One pass of every run over its own train rows. `order` [max size, R] holds row indices into `data`.

        `sizes` (host ints, descending) are the train rows of each run and `epoch` counts the finished epochs.
        Steps with a changing step size and steps of fewer than R runs run eagerly, the rest as CUDA graphs.
        Returns the dropout-mode predictions [max size, R]; entries past the size of a run are 0.
        """
        mask_fn = mask_fn or self.random_masks
        prot, lig, target = self.data
        self.order[: len(order)] = order
        self.pred.zero_()
        base = torch.tensor(sizes, device=self.device) * epoch + 1  # optimizer step count of each run at j = 0
        first = min(sizes) * epoch + 1
        j = 0
        for end in sorted(set(sizes)):
            k = sum(s >= end for s in sizes)
            while j < end:
                if self.use_graph and k == self.R and first + j > self.last_constant:
                    if self.graphs is None:
                        self._capture()
                    self.ptr.fill_(j)
                    while j < end:
                        count = GRAPH_STEPS if end - j >= GRAPH_STEPS else 1
                        self.mask_ptr.zero_()
                        self.mask_buf[:count] = mask_fn(count, k)
                        self.graphs[count].replay()
                        j += count
                    continue
                idx = self.order[j, :k]
                step = self.constant_step[:k] if first + j > self.last_constant else self.step_sizes(base[:k] + j)
                self.pred[j, :k] = self.train_step(prot[idx], lig[idx], mask_fn(1, k)[0], target[idx], step)
                j += 1
        return self.pred


HEAD = dict(expert_dim=768, expert_layers=1, num_experts=4, ple_layers=1, dropout=0.3, tower_layers=2, tower_hidden=128, tower_dropout=0.0)
COLUMNS = ("sequence", "smiles", "log10_value")


def draw_epoch(rng_state: torch.Tensor, n_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row order of the next epoch and the CPU random state after it. The draws are those of one epoch of the single-run
    script: the order, then one draw each for the validation and the test loader."""
    torch.set_rng_state(rng_state)
    order = epoch_order(n_rows)
    draw_loader_seed()
    draw_loader_seed()
    return order, torch.get_rng_state()


def is_complete(out_dir: Path, settings: dict) -> bool:
    settings_path = Path(out_dir) / "run_settings.json"
    return (Path(out_dir) / "final_results_test.csv").exists() and settings_path.exists() and json.loads(settings_path.read_text()) == settings


def train_group(runs: list[dict], embeddings_dir: str, device="cuda:0", epochs=100, limit_rows=0, use_graph=True) -> None:
    """Train the runs together. Each run is {"job", "seed", "out_dir", "overwrite"} and gets the files of train_single_split.py.

    Every run starts from the weights `set_seed(seed)` and CachedMPEKRegressor give, and draws its epoch orders from the
    CPU random stream of the single-run script. Dropout masks come from one device generator, reseeded at each epoch.
    """
    device = torch.device(device)
    datasets = {}
    for run in runs:
        job = run["job"]
        if job["train_path"] not in datasets:
            train = CachedMPEKDataset(job["train_path"], embeddings_dir, *COLUMNS, limit_rows=limit_rows)
            scaling = {"target_mean": train.target_mean, "target_std": train.target_std, "limit_rows": limit_rows}
            datasets[job["train_path"]] = (train, *(CachedMPEKDataset(job[f"{name}_path"], embeddings_dir, *COLUMNS, **scaling) for name in ("val", "test")))
    runs = sorted(runs, key=lambda run: -len(datasets[run["job"]["train_path"]][0]))
    sizes = [len(datasets[run["job"]["train_path"]][0]) for run in runs]
    offsets, trains = {}, []
    for path, (train, _, _) in datasets.items():
        offsets[path] = sum(len(t) for t in trains)
        trains.append(train)
    data = tuple(torch.cat([getattr(t, name) for t in trains]).to(device) for name in ("protein", "ligand", "target"))
    on_device = {path: tuple((ds.protein.to(device), ds.ligand.to(device)) for ds in splits[1:]) for path, splits in datasets.items()}

    models = []
    for run in runs:
        set_seed(run["seed"])
        models.append(CachedMPEKRegressor(**HEAD))
        run["rng"] = torch.get_rng_state()
        run["config"] = {**{k: v for k, v in run["job"].items()}, "seed": run["seed"], "epochs": epochs, "limit_rows": limit_rows, "group_size": len(runs), **HEAD}
        Path(run["out_dir"]).mkdir(parents=True, exist_ok=True)
        save_json(Path(run["out_dir"]) / "run_config.json", run["config"])
    stack = StackedMPEK(models, device, dropout=HEAD["dropout"], data=data, max_rows=max(sizes), use_graph=use_graph and device.type == "cuda")
    optimizers = [ChildTuningAdamW(m.parameters(), lr=stack.lr, betas=stack.betas, weight_decay=stack.weight_decay) for m in models]

    start_epoch, histories = 0, [[] for _ in runs]
    resumable = [Path(run["out_dir"]) / "checkpoint_last.pt" for run in runs]
    have = [p.exists() and not run["overwrite"] for p, run in zip(resumable, runs)]
    if any(have):
        if not all(have):
            raise RuntimeError("Only some runs of the group have a checkpoint. Rerun with --overwrite.")
        loaded = []
        for r, (path, run) in enumerate(zip(resumable, runs)):
            checkpoint = torch.load(path, map_location="cpu")
            models[r].load_state_dict(checkpoint["model_state_dict"])
            optimizers[r].load_state_dict(checkpoint["optimizer_state_dict"])
            stack.load_run(r, models[r], optimizers[r])
            histories[r] = pd.read_csv(Path(run["out_dir"]) / "epoch_metrics.csv").to_dict("records")
            loaded.append(int(checkpoint["epoch"]))
        if len(set(loaded)) != 1:
            raise RuntimeError(f"Checkpoints of the group are at different epochs {sorted(set(loaded))}. Rerun with --overwrite.")
        start_epoch = loaded[0]
        print(f"Resuming the group from epoch {start_epoch}", flush=True)

    def next_orders():
        """Orders of all runs for the next epoch, [max size, R]."""
        columns = []
        for run, n in zip(runs, sizes):
            order, run["rng"] = draw_epoch(run["rng"], n)
            columns.append(torch.nn.functional.pad(order + offsets[run["job"]["train_path"]], (0, max(sizes) - n)))
        return torch.stack(columns, 1).to(device)

    for _ in range(start_epoch):
        next_orders()
    started = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_started = time.time()
        order = next_orders()
        stack.generator.manual_seed(runs[0]["seed"] * 1000 + epoch)
        pred = stack.run_epoch(order, sizes, epoch - 1)
        val_rmse = []
        for r, run in enumerate(runs):
            path = run["job"]["train_path"]
            train, val, _ = datasets[path][:3]
            train_pred = torch.empty(sizes[r], device=device)
            train_pred[order[: sizes[r], r] - offsets[path]] = pred[: sizes[r], r]
            train_metrics = regression_metrics(train.raw_target, train_pred.cpu() * train.target_std + train.target_mean)
            val_metrics = regression_metrics(val.raw_target, stack.predict(r, *on_device[path][0]).cpu().squeeze(1) * val.target_std + val.target_mean)
            histories[r].append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
            pd.DataFrame(histories[r]).to_csv(Path(run["out_dir"]) / "epoch_metrics.csv", index=False)
            model, optimizer = models[r], optimizers[r]
            stack.store_run(r, model, optimizer, steps=epoch * sizes[r])
            latest = Path(run["out_dir"]) / "checkpoint_last.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": run["config"]}, str(latest) + ".tmp")
            Path(str(latest) + ".tmp").replace(latest)
            val_rmse.append(val_metrics["rmse"])
        print(f"epoch={epoch} {time.time() - epoch_started:.0f}s ({sum(sizes) / (time.time() - epoch_started):.0f} run-steps/s) mean val_rmse={sum(val_rmse) / len(runs):.6f}", flush=True)

    for r, run in enumerate(runs):
        path = run["job"]["train_path"]
        train, val, test = datasets[path]
        features = {"train": (data[0][offsets[path] : offsets[path] + sizes[r]], data[1][offsets[path] : offsets[path] + sizes[r]]), "val": on_device[path][0], "test": on_device[path][1]}
        for name, dataset in (("train", train), ("val", val), ("test", test)):
            y_pred = stack.predict(r, *features[name]).cpu().squeeze(1) * dataset.target_std + dataset.target_mean
            metrics = regression_metrics(dataset.raw_target, y_pred)
            metrics.update({**{k: run["job"][k] for k in ("task", "split_group", "split_name")}, "seed": run["seed"], "n": len(dataset), "rows_dropped": dataset.rows_dropped, "elapsed_seconds": time.time() - started})
            pd.DataFrame([metrics]).to_csv(Path(run["out_dir"]) / f"final_results_{name}.csv", index=False)
            pd.DataFrame({"y_true": dataset.raw_target.numpy(), "y_pred": y_pred.numpy()}).to_csv(Path(run["out_dir"]) / f"predictions_{name}.csv", index=False)
        save_json(Path(run["out_dir"]) / "run_settings.json", {"epochs": epochs, "limit_rows": limit_rows})
