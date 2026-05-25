#!/usr/bin/env python3
"""cycle_aware_attention: standalone no-GAP-leakage Pancake Graph n=12 experiment.

Fair-learning contract:
- the learned model never receives GAP counts, GAP deltas, adjacency-correctness bits,
  or any feature based on abs(x[i+1] - x[i]) == 1
- GAP is only used inside evaluation as a separate beam-search baseline
- hyperparameters are direct constants for Kaggle/Jupyter/Colab debugging
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Directly editable hyperparameters.
# -----------------------------------------------------------------------------
EXPERIMENT_NAME = "cycle_aware_attention"
N = 12
SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True
USE_TORCH_COMPILE = False

SAMPLED_ACTIONS = 8
D_MODEL = 128
N_HEADS = 4
ENCODER_LAYERS = 2
DROPOUT = 0.10
PLANNING_STEPS = 4

REPLAY_CAPACITY = 100_000
REPLAY_WARMUP = 4_096
FRESH_STATES_PER_STEP = 256
TRAIN_STEPS = 200
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
EMA_DECAY = 0.995
SOFT_TAU = 0.25
GAMMA = 1.0
SCRAMBLE_MIN = 1
SCRAMBLE_MAX = 40

BEAM_WIDTH = 64
BEAM_MAX_DEPTH = 80
EVAL_EPISODES = 64
EVAL_SCRAMBLE_MIN = 10
EVAL_SCRAMBLE_MAX = 50
PRINT_EVERY = 25
CONTRASTIVE_PRETRAIN_STEPS = 0
TDA_LOCAL_ACTIONS = 8
TDA_USE_TWO_HOP = True
# Paper methodology modes for TDA: pure, topology-aware, oracle-assisted.
TDA_MODE = "topology-aware"

# -----------------------------------------------------------------------------
# Reproducibility and metrics.
# -----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


@dataclass
class TrainMetrics:
    train_loss: float
    bellman_residual: float
    samples_per_sec: float
    gpu_peak_mb: float


@dataclass
class BeamMetrics:
    success_rate: float
    average_solution_length: float
    average_nodes_expanded: float
    runtime_sec: float

# -----------------------------------------------------------------------------
# Shared Pancake environment. No full graph materialization.
# -----------------------------------------------------------------------------
def goal_state(n: int = N, device: str | torch.device = DEVICE) -> torch.Tensor:
    return torch.arange(1, n + 1, dtype=torch.long, device=device)


def normalize_states(states: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(states, dtype=torch.long)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.numel() and int(x.min().item()) == 0:
        x = x + 1
    return x.clamp(1, x.shape[-1])


def apply_flip(states: torch.Tensor, actions: torch.Tensor | int) -> torch.Tensor:
    x = normalize_states(states)
    if isinstance(actions, int):
        actions = torch.full((x.shape[0],), actions, dtype=torch.long, device=x.device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=x.device).view(-1)
    if actions.numel() == 1 and x.shape[0] > 1:
        actions = actions.expand(x.shape[0])
    y = x.clone()
    for k in actions.unique(sorted=True).tolist():
        mask = actions == int(k)
        y[mask, : int(k)] = torch.flip(x[mask, : int(k)], dims=[1])
    return y


def sample_actions(batch: int, n: int = N, m: int = SAMPLED_ACTIONS, *, stochastic: bool, device: str | torch.device = DEVICE, seed: int | None = None) -> torch.Tensor:
    legal = torch.arange(2, n + 1, dtype=torch.long, device=device)
    m = min(m, legal.numel())
    if not stochastic:
        idx = torch.linspace(0, legal.numel() - 1, m, device=device).round().long()
        return legal[idx].expand(batch, -1)
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    scores = torch.rand(batch, legal.numel(), device=device, generator=gen)
    idx = scores.topk(m, dim=1).indices
    return legal[idx]


def sampled_neighbors(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    x = normalize_states(states)
    b, m = actions.shape
    flat_states = x[:, None, :].expand(b, m, x.shape[-1]).reshape(b * m, x.shape[-1])
    return apply_flip(flat_states, actions.reshape(-1)).reshape(b, m, x.shape[-1])


def scramble_batch(batch: int, depth_min: int, depth_max: int, *, device: str | torch.device = DEVICE) -> torch.Tensor:
    states = goal_state(device=device).expand(batch, N).clone()
    depths = torch.randint(depth_min, depth_max + 1, (batch,), device=device)
    prev = torch.zeros(batch, dtype=torch.long, device=device)
    for step in range(int(depths.max().item())):
        active = depths > step
        actions = torch.randint(2, N + 1, (batch,), dtype=torch.long, device=device)
        backtrack = actions == prev
        actions[backtrack] = 2 + ((actions[backtrack] - 1) % (N - 1))
        states[active] = apply_flip(states[active], actions[active])
        prev[active] = actions[active]
    return states


def random_walk_from(states: torch.Tensor, min_steps: int, max_steps: int, *, device: str | torch.device = DEVICE) -> torch.Tensor:
    """Sample states at controlled prefix-flip graph distances from anchors."""
    x = normalize_states(states).to(device).clone()
    batch = x.shape[0]
    depths = torch.randint(min_steps, max_steps + 1, (batch,), device=device)
    prev = torch.zeros(batch, dtype=torch.long, device=device)
    for step in range(int(depths.max().item())):
        active = depths > step
        actions = torch.randint(2, N + 1, (batch,), dtype=torch.long, device=device)
        backtrack = actions == prev
        actions[backtrack] = 2 + ((actions[backtrack] - 1) % (N - 1))
        x[active] = apply_flip(x[active], actions[active])
        prev[active] = actions[active]
    return x


def state_key(state: torch.Tensor | Iterable[int]) -> tuple[int, ...]:
    if isinstance(state, torch.Tensor):
        return tuple(int(v) for v in state.detach().cpu().tolist())
    return tuple(int(v) for v in state)


def gap_baseline_value(states: torch.Tensor) -> torch.Tensor:
    """GAP heuristic baseline, used only by evaluation beam_search(heuristic='gap')."""
    x = normalize_states(states)
    sentinel = torch.full((x.shape[0], 1), x.shape[-1] + 1, dtype=x.dtype, device=x.device)
    padded = torch.cat([x, sentinel], dim=1)
    return (torch.abs(padded[:, 1:] - padded[:, :-1]) != 1).sum(dim=1).float()


def edge_features(states: torch.Tensor, neighbors: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Action/transition features with no GAP, no adjacency correctness, no heuristic delta."""
    b, m, n = neighbors.shape
    repeated = states[:, None, :].expand(b, m, n).reshape(b * m, n).float()
    flat_neighbors = neighbors.reshape(b * m, n).float()
    flat_actions = actions.reshape(-1).float()
    positions = torch.arange(1, n + 1, device=states.device).float().view(1, n)
    cur_displacement = (repeated - positions).abs().mean(dim=1) / n
    nxt_displacement = (flat_neighbors - positions).abs().mean(dim=1) / n
    moved_mass = []
    prefix_spread = []
    for row, k in zip(repeated, flat_actions):
        kk = int(k.item())
        prefix = row[:kk]
        moved_mass.append(prefix.numel() / n)
        prefix_spread.append(prefix.float().std(unbiased=False) / n)
    moved_mass_t = torch.tensor(moved_mass, dtype=torch.float32, device=states.device)
    prefix_spread_t = torch.stack(prefix_spread)
    feats = torch.stack([
        flat_actions / n,
        moved_mass_t,
        nxt_displacement - cur_displacement,
        prefix_spread_t,
        (flat_actions % 2) / 1.0,
    ], dim=-1)
    return feats.view(b, m, 5)

# -----------------------------------------------------------------------------
# Replay buffer: CPU storage, uniform sampling. No prioritization and no GAP.
# -----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAPACITY, seed: int = SEED) -> None:
        self.capacity = int(capacity)
        self.states = torch.empty((self.capacity, N), dtype=torch.long)
        self.size = 0
        self.pos = 0
        self.rng = torch.Generator().manual_seed(seed)

    def add(self, states: torch.Tensor) -> None:
        x = states.detach().cpu().long().view(-1, N)
        n = x.shape[0]
        if n >= self.capacity:
            self.states.copy_(x[-self.capacity:])
            self.size = self.capacity
            self.pos = 0
            return
        end = self.pos + n
        if end <= self.capacity:
            self.states[self.pos:end].copy_(x)
        else:
            first = self.capacity - self.pos
            self.states[self.pos:].copy_(x[:first])
            self.states[: end % self.capacity].copy_(x[first:])
        self.pos = end % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int, device: str | torch.device = DEVICE) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("ReplayBuffer is empty")
        idx = torch.randint(0, self.size, (batch_size,), generator=self.rng)
        return self.states[idx].to(device, non_blocking=True)

# -----------------------------------------------------------------------------
# Real local TDA. This is only injected for tda_persistent_homology.
# -----------------------------------------------------------------------------
def _kendall_tau_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Kendall tau distance between two pancake permutations."""
    pos_in_b = {int(value): idx for idx, value in enumerate(b.tolist())}
    order = [pos_in_b[int(value)] for value in a.tolist()]
    inv = 0
    for i in range(len(order)):
        oi = order[i]
        for j in range(i + 1, len(order)):
            inv += int(oi > order[j])
    denom = max(len(order) * (len(order) - 1) / 2, 1.0)
    return float(inv / denom)


def _breakpoint_edge_set(p: np.ndarray) -> set[tuple[int, int]]:
    """Undirected local adjacency set used only in topology-aware/oracle TDA modes."""
    padded = [0] + [int(v) for v in p.tolist()] + [N + 1]
    return {tuple(sorted((padded[i], padded[i + 1]))) for i in range(len(padded) - 1)}


def _breakpoint_edge_distance(a: np.ndarray, b: np.ndarray) -> float:
    ea = _breakpoint_edge_set(a)
    eb = _breakpoint_edge_set(b)
    return float(len(ea.symmetric_difference(eb)) / max(len(ea.union(eb)), 1))


def _gap_count_np(p: np.ndarray) -> int:
    padded = np.concatenate([p.astype(np.int64), np.asarray([N + 1], dtype=np.int64)])
    return int(np.sum(np.abs(np.diff(padded)) != 1))


def _oracle_gap_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(_gap_count_np(a) - _gap_count_np(b)) / N)


def _local_flip_distance(points: np.ndarray) -> np.ndarray:
    """Exact shortest prefix-flip distances inside the sampled local point cloud."""
    keys = [tuple(int(v) for v in row.tolist()) for row in points]
    index = {key: i for i, key in enumerate(keys)}
    size = len(keys)
    graph = [[] for _ in range(size)]
    for i, key in enumerate(keys):
        state = torch.tensor(key, dtype=torch.long).view(1, N)
        actions = torch.arange(2, N + 1, dtype=torch.long).view(1, -1)
        nbrs = sampled_neighbors(state, actions).reshape(-1, N)
        for nbr in nbrs:
            j = index.get(state_key(nbr))
            if j is not None:
                graph[i].append(j)
    distances = np.full((size, size), fill_value=float(size), dtype=np.float64)
    for src in range(size):
        distances[src, src] = 0.0
        queue = [src]
        for u in queue:
            for v in graph[u]:
                if distances[src, v] == float(size):
                    distances[src, v] = distances[src, u] + 1.0
                    queue.append(v)
    return distances / max(float(distances[np.isfinite(distances)].max()), 1.0)


def _pairwise_pancake_metric(points: np.ndarray, mode: str = TDA_MODE) -> np.ndarray:
    """Mode-specific metric space for Vietoris-Rips TDA; uses Pancake-aware metrics instead of coordinate mismatch distance."""
    size = points.shape[0]
    kendall = np.zeros((size, size), dtype=np.float64)
    breakpoint = np.zeros_like(kendall)
    oracle_gap = np.zeros_like(kendall)
    for i in range(size):
        for j in range(i + 1, size):
            kendall[i, j] = kendall[j, i] = _kendall_tau_distance(points[i], points[j])
            if mode in {"topology-aware", "oracle-assisted"}:
                breakpoint[i, j] = breakpoint[j, i] = _breakpoint_edge_distance(points[i], points[j])
            if mode == "oracle-assisted":
                oracle_gap[i, j] = oracle_gap[j, i] = _oracle_gap_distance(points[i], points[j])
    flip = _local_flip_distance(points)
    if mode == "pure":
        metric = 0.65 * kendall + 0.35 * flip
    elif mode == "topology-aware":
        metric = 0.45 * kendall + 0.25 * flip + 0.30 * breakpoint
    elif mode == "oracle-assisted":
        metric = 0.30 * kendall + 0.20 * flip + 0.25 * breakpoint + 0.25 * oracle_gap
    else:
        raise ValueError(f"Unknown TDA_MODE={mode!r}; use pure, topology-aware, or oracle-assisted")
    np.fill_diagonal(metric, 0.0)
    return np.maximum(metric, metric.T).astype(np.float64)


def _local_points_for_tda(state: tuple[int, ...]) -> np.ndarray:
    base = torch.tensor(state, dtype=torch.long).view(1, N)
    actions = sample_actions(1, m=TDA_LOCAL_ACTIONS, stochastic=False, device="cpu")
    one_hop = sampled_neighbors(base, actions).reshape(-1, N)
    points = [base.squeeze(0)] + [row for row in one_hop]
    if TDA_USE_TWO_HOP:
        second_actions = sample_actions(one_hop.shape[0], m=min(4, SAMPLED_ACTIONS), stochastic=False, device="cpu")
        two_hop = sampled_neighbors(one_hop, second_actions).reshape(-1, N)
        # Keep local computation bounded and deterministic.
        for row in two_hop[:24]:
            points.append(row)
    unique = list(dict.fromkeys(state_key(p) for p in points))
    return np.asarray(unique, dtype=np.int64)


def _diagram_stats(diagrams: list[np.ndarray]) -> np.ndarray:
    """Compact TDA vector: Betti counts, entropy, max persistence, mean persistence."""
    feats: list[float] = []
    for dim in range(2):
        dgm = diagrams[dim] if dim < len(diagrams) else np.empty((0, 2), dtype=np.float64)
        if dgm.size == 0:
            lifetimes = np.zeros(0, dtype=np.float64)
        else:
            finite = dgm[np.isfinite(dgm[:, 1])]
            lifetimes = np.maximum(finite[:, 1] - finite[:, 0], 0.0) if finite.size else np.zeros(0, dtype=np.float64)
        total = float(lifetimes.sum())
        entropy = 0.0
        if lifetimes.size and total > 0:
            probs = lifetimes / total
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        feats.extend([
            float(lifetimes.size),
            entropy,
            float(lifetimes.max()) if lifetimes.size else 0.0,
            float(lifetimes.mean()) if lifetimes.size else 0.0,
        ])
    return np.asarray(feats, dtype=np.float32)


@lru_cache(maxsize=20_000)
def tda_features_cached(state: tuple[int, ...], mode: str = TDA_MODE) -> tuple[float, ...]:
    points = _local_points_for_tda(state)
    distances = _pairwise_pancake_metric(points, mode=mode)
    try:
        from ripser import ripser  # type: ignore
        diagrams = ripser(distances, distance_matrix=True, maxdim=1)["dgms"]
        feats = _diagram_stats(diagrams)
    except ModuleNotFoundError:
        try:
            import gudhi as gd  # type: ignore
            rips = gd.RipsComplex(distance_matrix=distances.tolist(), max_edge_length=float(distances.max()))
            st = rips.create_simplex_tree(max_dimension=2)
            st.persistence()
            diagrams = []
            for dim in range(2):
                pairs = st.persistence_intervals_in_dimension(dim)
                diagrams.append(np.asarray(pairs, dtype=np.float64).reshape(-1, 2) if len(pairs) else np.empty((0, 2), dtype=np.float64))
            feats = _diagram_stats(diagrams)
        except ModuleNotFoundError as exc:
            raise RuntimeError("tda_persistent_homology requires `pip install ripser` or `pip install gudhi`.") from exc
    return tuple(feats.tolist())


def tda_feature_tensor(states: torch.Tensor) -> torch.Tensor:
    rows = [tda_features_cached(state_key(s), TDA_MODE) for s in states.detach().cpu()]
    return torch.tensor(rows, dtype=torch.float32, device=states.device)

# -----------------------------------------------------------------------------
# Value model. Default mode has no GAP channel; oracle-assisted TDA is opt-in.
# -----------------------------------------------------------------------------
class PermutationEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value_emb = nn.Embedding(N + 2, D_MODEL)
        self.pos_emb = nn.Embedding(N, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=4 * D_MODEL,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, ENCODER_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, states: torch.Tensor, token_bias: torch.Tensor | None = None) -> torch.Tensor:
        b, n = states.shape
        pos = torch.arange(n, device=states.device).expand(b, n)
        x = self.value_emb(states.clamp(1, N + 1)) + self.pos_emb(pos)
        if token_bias is not None:
            x = x + token_bias
        return self.norm(self.encoder(x).mean(dim=1))


class PancakeValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = PermutationEncoder()
        self.relative_proj = nn.Linear(4, D_MODEL)
        self.edge_proj = nn.Sequential(nn.Linear(5, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.action_emb = nn.Embedding(N + 1, D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True)
        self.planner = nn.GRUCell(D_MODEL, D_MODEL)
        self.history_proj = nn.Linear(3, D_MODEL)
        self.lap_proj = nn.Linear(4, D_MODEL)
        self.tda_proj = nn.Linear(8, D_MODEL)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * D_MODEL),
            nn.Linear(2 * D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 1),
            nn.Softplus(),
        )
        self.last_attention: torch.Tensor | None = None

    def relative_token_bias(self, states: torch.Tensor) -> torch.Tensor:
        # Relative geometry without adjacency-correctness or GAP-style oracle features.
        pos = torch.arange(1, N + 1, device=states.device).float().view(1, N)
        vals = states.float()
        displacement = (vals - pos) / N
        centered_value = (vals - vals.mean(dim=1, keepdim=True)) / N
        rank_fraction = vals / N
        prefix_mean = torch.cumsum(vals, dim=1) / torch.arange(1, N + 1, device=states.device).float().view(1, N) / N
        return self.relative_proj(torch.stack([displacement, centered_value, rank_fraction, prefix_mean], dim=-1))

    def laplacian_features(self, states: torch.Tensor) -> torch.Tensor:
        x = states.float()
        return torch.stack([x.mean(1) / N, x.std(1) / N, (x[:, 0] - 1).abs() / N, (x[:, -1] - N).abs() / N], dim=-1)

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        bias = self.relative_token_bias(states) if EXPERIMENT_NAME in {"relative_flip_positional_encoding", "permutation_equivariant_transformer"} else None
        return self.encoder(states, bias)

    def forward(self, states: torch.Tensor, actions: torch.Tensor | None = None, history_features: torch.Tensor | None = None) -> torch.Tensor:
        states = normalize_states(states)
        b = states.shape[0]
        if actions is None:
            actions = sample_actions(b, stochastic=False, device=states.device)
        neighbors = sampled_neighbors(states, actions)
        m = neighbors.shape[1]

        current = self.encode(states)
        neighbor_emb = self.encode(neighbors.reshape(b * m, N)).view(b, m, D_MODEL)

        if EXPERIMENT_NAME in {"edge_conditioned_action_attention", "cycle_aware_attention", "sparse_graph_transformer"}:
            neighbor_emb = neighbor_emb + self.edge_proj(edge_features(states, neighbors, actions)) + self.action_emb(actions)
        if EXPERIMENT_NAME == "relative_flip_positional_encoding":
            neighbor_emb = neighbor_emb + self.action_emb(actions)
        if EXPERIMENT_NAME == "laplacian_positional_encoding":
            current = current + self.lap_proj(self.laplacian_features(states))
        if EXPERIMENT_NAME == "tda_persistent_homology":
            current = current + self.tda_proj(tda_feature_tensor(states))
        if EXPERIMENT_NAME == "cycle_aware_attention" and history_features is not None:
            current = current + self.history_proj(history_features.float())

        q = self.q(current).unsqueeze(1)
        k = self.k(neighbor_emb)
        v = self.v(neighbor_emb)
        if EXPERIMENT_NAME == "sparse_graph_transformer":
            scores = (q @ k.transpose(1, 2)) / math.sqrt(D_MODEL)
            keep = min(4, m)
            keep_idx = scores.topk(keep, dim=-1).indices
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(-1, keep_idx, 0.0)
            weights = (scores + mask).softmax(dim=-1)
            context = weights @ v
            self.last_attention = weights.detach()
        else:
            context, weights = self.attn(q, k, v, need_weights=True)
            self.last_attention = weights.detach()
        context = context.squeeze(1)

        if EXPERIMENT_NAME == "neural_algorithmic_reasoning":
            hidden = current
            for _ in range(PLANNING_STEPS):
                hidden = self.planner(context, hidden)
            current = hidden

        return self.head(torch.cat([current, context], dim=-1)).squeeze(-1)

# -----------------------------------------------------------------------------
# AVI target, optional pretraining, train loop. No GAP in targets or losses.
# -----------------------------------------------------------------------------
@torch.no_grad()
def soft_bellman_target(target_net: nn.Module, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    b, m = actions.shape
    neighbors = sampled_neighbors(states, actions).reshape(b * m, N)
    next_actions = sample_actions(b * m, stochastic=False, device=states.device)
    with torch.autocast(device_type="cuda", enabled=USE_AMP and states.is_cuda):
        next_v = target_net(neighbors, next_actions).view(b, m)
        soft_min = -SOFT_TAU * torch.logsumexp(-next_v / max(SOFT_TAU, 1e-6), dim=1)
        y = 1.0 + GAMMA * soft_min
        is_goal = (states == goal_state(device=states.device)).all(dim=1)
        return torch.where(is_goal, torch.zeros_like(y), y).clamp_min(0.0)


@torch.no_grad()
def ema_update(model: nn.Module, target_net: nn.Module) -> None:
    for target_param, param in zip(target_net.parameters(), model.parameters()):
        target_param.data.mul_(EMA_DECAY).add_(param.data, alpha=1.0 - EMA_DECAY)


def contrastive_pretrain(model: PancakeValueNet, optimizer: torch.optim.Optimizer) -> None:
    if CONTRASTIVE_PRETRAIN_STEPS <= 0:
        return
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE == "cuda")
    model.train()
    for step in range(1, CONTRASTIVE_PRETRAIN_STEPS + 1):
        anchors = scramble_batch(BATCH_SIZE, 4, 40, device=DEVICE)
        # Graph metric learning: positives are guaranteed local-neighborhood states
        # (true prefix-flip graph distance <= 2), while negatives are long random
        # walks from the same anchors and are therefore far under the graph metric.
        positives = random_walk_from(anchors, min_steps=1, max_steps=2, device=DEVICE)
        negatives = random_walk_from(anchors, min_steps=18, max_steps=48, device=DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=USE_AMP and DEVICE == "cuda"):
            za = F.normalize(model.encode(anchors), dim=-1)
            zp = F.normalize(model.encode(positives), dim=-1)
            zn = F.normalize(model.encode(negatives), dim=-1)
            logits = torch.cat([(za * zp).sum(-1, keepdim=True), za @ zn.T], dim=1) / 0.1
            loss = F.cross_entropy(logits, torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == CONTRASTIVE_PRETRAIN_STEPS:
            print(f"pretrain_loss={float(loss.detach().cpu()):.4f}")


def train(model: PancakeValueNet) -> TrainMetrics:
    target_net = PancakeValueNet().to(DEVICE)
    target_net.load_state_dict(model.state_dict())
    target_net.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    contrastive_pretrain(model, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE == "cuda")
    replay = ReplayBuffer()
    replay.add(scramble_batch(REPLAY_WARMUP, 1, SCRAMBLE_MAX, device="cpu"))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    total_samples = 0
    last = TrainMetrics(float("nan"), float("nan"), 0.0, 0.0)

    for step in range(1, TRAIN_STEPS + 1):
        replay.add(scramble_batch(FRESH_STATES_PER_STEP, SCRAMBLE_MIN, SCRAMBLE_MAX, device="cpu"))
        states = replay.sample(BATCH_SIZE, DEVICE)
        actions = sample_actions(BATCH_SIZE, stochastic=True, device=DEVICE, seed=SEED + step)
        targets = soft_bellman_target(target_net, states, actions)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=USE_AMP and DEVICE == "cuda"):
            pred = model(states, actions)
            loss = F.smooth_l1_loss(pred, targets)
            residual = (pred.detach() - targets).abs().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        ema_update(model, target_net)

        total_samples += BATCH_SIZE
        elapsed = max(time.perf_counter() - start_time, 1e-9)
        last = TrainMetrics(
            train_loss=float(loss.detach().cpu()),
            bellman_residual=float(residual.detach().cpu()),
            samples_per_sec=total_samples / elapsed,
            gpu_peak_mb=gpu_memory_mb(),
        )
        if step == 1 or step % PRINT_EVERY == 0 or step == TRAIN_STEPS:
            print(
                f"step={step:04d} train_loss={last.train_loss:.4f} "
                f"bellman_residual={last.bellman_residual:.4f} "
                f"samples/sec={last.samples_per_sec:.1f} gpu_peak_mb={last.gpu_peak_mb:.1f}"
            )
    return last

# -----------------------------------------------------------------------------
# Evaluation: learned model beam, GAP baseline beam, optional random baseline.
# GAP is called only inside heuristic == 'gap'.
# -----------------------------------------------------------------------------
@torch.no_grad()
def beam_search(model: PancakeValueNet | None, start: torch.Tensor, *, heuristic: Literal["model", "gap", "random"] = "model") -> dict[str, float | bool]:
    goal = state_key(goal_state(device="cpu"))
    start_key = state_key(start)
    beam: list[tuple[float, tuple[int, ...], list[int]]] = [(0.0, start_key, [])]
    visited: dict[tuple[int, ...], int] = {start_key: 1}
    expanded = 0
    t0 = time.perf_counter()

    for _depth in range(BEAM_MAX_DEPTH + 1):
        solved = [path for _, key, path in beam if key == goal]
        if solved:
            return {"success": True, "solution_length": float(len(solved[0])), "nodes_expanded": float(expanded), "runtime_sec": time.perf_counter() - t0}
        states = torch.tensor([key for _, key, _ in beam], dtype=torch.long, device=DEVICE)
        actions = sample_actions(states.shape[0], stochastic=False, device=DEVICE)
        flat_states = states[:, None, :].expand(-1, actions.shape[1], -1).reshape(-1, N)
        next_states = apply_flip(flat_states, actions.reshape(-1))
        if heuristic == "model":
            assert model is not None
            next_actions = sample_actions(next_states.shape[0], stochastic=False, device=DEVICE)
            values = model(next_states, next_actions).detach().float().cpu()
        elif heuristic == "gap":
            values = gap_baseline_value(next_states).detach().float().cpu()
        else:
            values = torch.rand(next_states.shape[0])

        candidates: list[tuple[float, tuple[int, ...], list[int]]] = []
        flat_actions = actions.reshape(-1).detach().cpu()
        next_states_cpu = next_states.detach().cpu()
        for i, ns in enumerate(next_states_cpu):
            parent = i // actions.shape[1]
            key = state_key(ns)
            action = int(flat_actions[i].item())
            revisit_penalty = visited.get(key, 0) * (2.0 if EXPERIMENT_NAME == "cycle_aware_attention" else 0.0)
            path = beam[parent][2] + [action]
            score = len(path) + float(values[i]) + revisit_penalty
            candidates.append((score, key, path))
            visited[key] = visited.get(key, 0) + 1
        expanded += len(beam)
        candidates.sort(key=lambda item: item[0])
        beam = candidates[:BEAM_WIDTH]
    return {"success": False, "solution_length": float("nan"), "nodes_expanded": float(expanded), "runtime_sec": time.perf_counter() - t0}


def summarize_beam(results: list[dict[str, float | bool]]) -> BeamMetrics:
    successes = [r for r in results if bool(r["success"])]
    return BeamMetrics(
        success_rate=len(successes) / max(len(results), 1),
        average_solution_length=float(np.mean([r["solution_length"] for r in successes])) if successes else float("nan"),
        average_nodes_expanded=float(np.mean([r["nodes_expanded"] for r in results])),
        runtime_sec=float(np.sum([r["runtime_sec"] for r in results])),
    )


def evaluate(model: PancakeValueNet) -> dict[str, BeamMetrics]:
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    starts = scramble_batch(EVAL_EPISODES, EVAL_SCRAMBLE_MIN, EVAL_SCRAMBLE_MAX, device="cpu")
    summaries: dict[str, BeamMetrics] = {}
    for heuristic in ("model", "gap", "random"):
        results = [beam_search(model if heuristic == "model" else None, s, heuristic=heuristic) for s in starts]
        summaries[heuristic] = summarize_beam(results)
        m = summaries[heuristic]
        print(
            f"heuristic={heuristic} success_rate={m.success_rate:.3f} "
            f"average_solution_length={m.average_solution_length:.2f} "
            f"nodes_expanded={m.average_nodes_expanded:.1f} runtime_sec={m.runtime_sec:.2f}"
        )
    print(f"eval_gpu_peak_mb={gpu_memory_mb():.1f}")
    return summaries


def main() -> None:
    print(f"experiment={EXPERIMENT_NAME} device={DEVICE} n={N} sampled_actions={SAMPLED_ACTIONS}")
    set_seed(SEED)
    model = PancakeValueNet().to(DEVICE)
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)
    train_metrics = train(model)
    eval_metrics = evaluate(model)
    model_metrics = eval_metrics["model"]
    gap_metrics = eval_metrics["gap"]
    print(
        f"FINAL train_loss={train_metrics.train_loss:.4f} "
        f"beam_success_rate={model_metrics.success_rate:.3f} "
        f"average_solution_length={model_metrics.average_solution_length:.2f} "
        f"gap_success_rate={gap_metrics.success_rate:.3f} "
        f"gap_average_solution_length={gap_metrics.average_solution_length:.2f}"
    )


if __name__ == "__main__":
    main()
