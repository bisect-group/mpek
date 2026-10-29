"""Exact speed-ups for batch-1 training: a multi-tensor ChildTuningAdamW and CUDA graphs.

Both give the same losses, weights and predictions, bit for bit, as the plain loop.
"""

import math

import numpy as np
import torch

from MTLKcatKM.utils import ChildTuningAdamW

GRAPH_STEPS = 50


class ForeachChildTuningAdamW(ChildTuningAdamW):
    """ChildTuningAdamW (mode off) with the same maths in multi-tensor kernels."""

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            for p in params:
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"], state["exp_avg_sq"] = torch.zeros_like(p), torch.zeros_like(p)
                state["step"] += 1
            grads = [p.grad for p in params]
            exp_avg = [self.state[p]["exp_avg"] for p in params]
            exp_avg_sq = [self.state[p]["exp_avg_sq"] for p in params]
            beta1, beta2 = group["betas"]
            t = self.state[params[0]]["step"]
            torch._foreach_mul_(exp_avg, beta1)
            torch._foreach_add_(exp_avg, grads, alpha=1.0 - beta1)
            torch._foreach_mul_(exp_avg_sq, beta2)
            torch._foreach_addcmul_(exp_avg_sq, grads, grads, value=1.0 - beta2)
            denom = torch._foreach_sqrt(exp_avg_sq)
            torch._foreach_add_(denom, group["eps"])
            step_size = group["lr"] * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
            torch._foreach_addcdiv_(params, exp_avg, denom, value=-step_size)
            torch._foreach_add_(params, params, alpha=-group["lr"] * group["weight_decay"])


def constant_step_after(lr: float, beta1: float = 0.9, beta2: float = 0.999) -> int:
    """Last step t whose bias-corrected step size differs in fp32 from the limit. Later steps use one fp32 value."""
    t = np.arange(1, 200_001, dtype=np.float64)
    step_size = (lr * np.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)).astype(np.float32)
    return int(t[step_size != np.float32(lr)].max())


def optimizer_steps(optimizer) -> int:
    return max((s["step"] for s in optimizer.state.values() if s), default=0)


def _graph(body, count):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):
            body()
    return graph


def _warm_up(body):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


class GraphedTrainer:
    """Batch-1 epochs with every step (gather, forward, loss, backward, optimizer) in CUDA graphs.

    The bias-corrected step size of ChildTuningAdamW is a host constant in a graph. It has one fp32 value after
    `constant_step_after` steps, so earlier steps run eagerly and later steps replay the graph.
    """

    def __init__(self, model, optimizer, protein, ligand, target, lr: float, beta1: float = 0.9, beta2: float = 0.999):
        self.model, self.optimizer = model, optimizer
        self.protein, self.ligand, self.target = protein, ligand, target
        self.eager_until = constant_step_after(lr, beta1, beta2)
        self.order = torch.zeros(len(target), dtype=torch.long, device=target.device)
        self.ptr = torch.zeros(1, dtype=torch.long, device=target.device)
        self.train_pred = torch.zeros_like(target)
        self.graphs = {}

    def _update(self, idx):
        pred = self.model(self.protein.index_select(0, idx), self.ligand.index_select(0, idx)).float()
        loss = (pred - self.target.index_select(0, idx)).pow(2).sum() / (len(idx) + 1e-5)  # original compute_loss form
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.train_pred.index_copy_(0, idx, pred.detach())

    def _body(self):
        idx = self.order.index_select(0, self.ptr)
        self.ptr += 1
        self._update(idx)

    def _capture(self):
        """Capture without changing weights, optimizer state, step counts or RNG state."""
        params = [p for p in self.model.parameters() if self.optimizer.state[p]]
        tensors = [self.train_pred] + [t for p in params for t in (p, self.optimizer.state[p]["exp_avg"], self.optimizer.state[p]["exp_avg_sq"])]
        saved = [t.detach().clone() for t in tensors]
        steps = [self.optimizer.state[p]["step"] for p in params]
        rng = torch.cuda.get_rng_state()
        _warm_up(self._body)
        with torch.no_grad():
            for t, s in zip(tensors, saved):
                t.copy_(s)
        self.graphs = {count: _graph(self._body, count) for count in (GRAPH_STEPS, 1)}
        with torch.no_grad():
            for t, s in zip(tensors, saved):
                t.copy_(s)
        for p, step in zip(params, steps):
            self.optimizer.state[p]["step"] = step
        torch.cuda.set_rng_state(rng)

    def run(self, order: torch.Tensor) -> torch.Tensor:
        """Train over `order` (one row per step) and return the dropout-mode predictions per row."""
        self.model.train()
        self.train_pred.zero_()
        eager = min(len(order), max(0, self.eager_until - optimizer_steps(self.optimizer)))
        for i in range(eager):
            self._update(order[i : i + 1])
        rest = len(order) - eager
        if rest:
            if not self.graphs:
                self._capture()
            self.order[:rest].copy_(order[eager:])
            self.ptr.zero_()
            for _ in range(rest // GRAPH_STEPS):
                self.graphs[GRAPH_STEPS].replay()
            for _ in range(rest % GRAPH_STEPS):
                self.graphs[1].replay()
            for state in self.optimizer.state.values():
                if state:
                    state["step"] += rest
        return self.train_pred


class GraphedPredictor:
    """Eval-mode predictions at batch size 1 for fixed features, in CUDA graphs."""

    def __init__(self, model, protein, ligand):
        self.model, self.protein, self.ligand = model, protein, ligand
        self.out = torch.zeros(len(protein), 1, device=protein.device)
        self.ptr = torch.zeros(1, dtype=torch.long, device=protein.device)
        was_training = model.training
        model.eval()
        _warm_up(self._body)
        self.graphs = {count: _graph(self._body, count) for count in (GRAPH_STEPS, 1)}
        model.train(was_training)

    @torch.no_grad()
    def _body(self):
        pred = self.model(self.protein.index_select(0, self.ptr), self.ligand.index_select(0, self.ptr))
        self.out.index_copy_(0, self.ptr, pred)
        self.ptr += 1

    def predict(self) -> torch.Tensor:
        n = len(self.out)
        self.ptr.zero_()
        for _ in range(n // GRAPH_STEPS):
            self.graphs[GRAPH_STEPS].replay()
        for _ in range(n % GRAPH_STEPS):
            self.graphs[1].replay()
        return self.out.clone()
