from __future__ import annotations

import math
import random
import time
import json
from collections import deque, Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

import cayleypy as cp
from cayleypy.predictor import Predictor
from torch_geometric.nn import AttentionalAggregation

HAS_PYG = True

from functools import lru_cache

# ============================================================================
# 🌍 Environment and graph primitives (0-indexed)
# ============================================================================

@lru_cache(maxsize=4096)
def _gap_heuristic_cached(state_tuple: tuple) -> float:
    """Кэшированная версия gap_heuristic."""
    state = torch.tensor([state_tuple], dtype=torch.long)
    return float(gap_heuristic(state).item())


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def goal_state(n: int, *, device: str | torch.device | None = None) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long, device=device)

def as_batched_long(
    states: torch.Tensor | list[int] | list[list[int]],
    *,
    device: str | torch.device | None = None,
    normalize: bool = True,
    n: int | None = None,
) -> torch.Tensor:
    x = torch.as_tensor(states, dtype=torch.long, device=device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if normalize and x.numel() > 0:
        if int(x.min().detach().cpu()) == 1:
            x = x - 1
        if n is None:
            n = x.shape[-1]
        x = x.clamp(0, n - 1)
    return x

def normalize_permutations(states: torch.Tensor, n: int | None = None) -> torch.Tensor:
    return as_batched_long(states, normalize=True, n=n)

def is_goal(states: torch.Tensor, n: int | None = None) -> torch.Tensor:
    x = as_batched_long(states, n=n)
    if n is None:
        n = x.shape[-1]
    return (x == goal_state(n, device=x.device).view(1, -1)).all(dim=1)

def apply_flip(states: torch.Tensor, k: int) -> torch.Tensor:
    x = as_batched_long(states)
    y = x.clone()
    y[:, :k] = torch.flip(x[:, :k], dims=[1])
    return y

def apply_flip_actions(
    states: torch.Tensor, 
    actions: torch.Tensor, 
    *, 
    normalize: bool = True
) -> torch.Tensor:
    x = as_batched_long(states, normalize=normalize)
    if actions.dim() == 1:
        actions = actions.view(1, -1).expand(x.shape[0], -1)
    actions = actions.to(x.device).long().clamp(2, x.shape[1])
    batch, n = x.shape
    n_actions = actions.shape[1]
    pos = torch.arange(n, device=x.device).view(1, 1, n)
    k = actions.unsqueeze(-1)
    gather_idx = torch.where(pos < k, k - 1 - pos, pos)
    x_expanded = x.unsqueeze(1)
    result = torch.gather(x_expanded.expand(-1, n_actions, -1), dim=2, index=gather_idx)
    return result

def all_flip_actions(batch_size: int, n: int, *, device: str | torch.device | None = None) -> torch.Tensor:
    return torch.arange(2, n + 1, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)

def stratified_flip_actions(batch_size: int, n: int, num_actions: int, *, device: str | torch.device | None = None) -> torch.Tensor:
    device = torch.device(device or "cpu")
    if n < 2 or num_actions <= 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=device)
    m = min(num_actions, n - 1)
    if m == n - 1:
        return all_flip_actions(batch_size, n, device=device)
    base = torch.linspace(2, n, m, device=device).round().long().unique()
    if base.numel() < m:
        extra = torch.arange(2, n + 1, device=device)
        base = torch.cat([base, extra]).unique()[:m]
    return base.view(1, -1).expand(batch_size, -1).contiguous()

def sample_flip_actions(
    batch_size: int,
    n: int,
    num_actions: int,
    *,
    device: str | torch.device | None = None,
    include_extremes: bool = True,
    policy_logits: torch.Tensor | None = None,
    policy_fraction: float = 0.5,
) -> torch.Tensor:
    device = torch.device(device or "cpu")
    if n < 2 or num_actions <= 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=device)
    m = min(num_actions, n - 1)
    if m == n - 1:
        return all_flip_actions(batch_size, n, device=device)
    out_parts: list[torch.Tensor] = []
    fixed: list[int] = []
    if include_extremes:
        fixed.extend([2, n])
    fixed.extend(stratified_flip_actions(1, n, min(4, m), device=device)[0].detach().cpu().tolist())
    fixed_tensor = torch.tensor(sorted({k for k in fixed if 2 <= k <= n}), dtype=torch.long, device=device)
    if fixed_tensor.numel() > 0:
        out_parts.append(fixed_tensor[:m].view(1, -1).expand(batch_size, -1))
    remaining = m - sum(part.shape[1] for part in out_parts)
    if remaining > 0 and policy_logits is not None:
        pol_m = min(remaining, max(1, int(round(m * policy_fraction))))
        probs = torch.softmax(policy_logits.detach().to(device), dim=-1)
        top = torch.multinomial(probs, pol_m, replacement=False) + 2
        out_parts.append(top)
        remaining -= pol_m
    if remaining > 0:
        out_parts.append(torch.randint(2, n + 1, (batch_size, remaining), dtype=torch.long, device=device))
    actions = torch.cat(out_parts, dim=1)[:, :m]
    if actions.shape[1] < m:
        pad = torch.randint(2, n + 1, (batch_size, m - actions.shape[1]), dtype=torch.long, device=device)
        actions = torch.cat([actions, pad], dim=1)
    return actions

def transition_sampled_neighbors(
    states: torch.Tensor,
    *,
    num_actions: int,
    actions: torch.Tensor | None = None,
    policy_logits: torch.Tensor | None = None,
    policy_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = as_batched_long(states)
    if actions is None:
        actions = sample_flip_actions(
            x.shape[0], x.shape[1], num_actions, 
            device=x.device, 
            policy_logits=policy_logits,
            policy_fraction=policy_fraction
        )
    return apply_flip_actions(x, actions, normalize=False), actions

def pancake_neighbors(states: torch.Tensor, *, max_neighbors: int | None = None, mode: str = "all") -> torch.Tensor:
    x = as_batched_long(states)
    batch, n = x.shape
    flips = torch.arange(2, n + 1, device=x.device)
    if max_neighbors is not None and max_neighbors < flips.numel():
        if mode == "random":
            idx = torch.randperm(flips.numel(), device=x.device)[:max_neighbors]
        elif mode in {"all", "sampled", "stratified"}:
            idx = torch.linspace(0, flips.numel() - 1, max_neighbors, device=x.device).round().long().unique()
        else:
            raise ValueError(f"Unknown neighbor mode: {mode}")
        flips = flips[idx]
    if flips.numel() == 0:
        return x.new_empty((batch, 0, n))
    return apply_flip_actions(x, flips, normalize=False)

def gap_heuristic(states: torch.Tensor) -> torch.Tensor:
    x = as_batched_long(states)
    n = x.shape[-1]
    sentinel = torch.full((x.shape[0], 1), n, dtype=x.dtype, device=x.device)
    padded = torch.cat([x, sentinel], dim=1)
    return (torch.abs(padded[:, 1:] - padded[:, :-1]) != 1).sum(dim=1).float()

def exact_bfs_shells(n: int, *, depth: int, max_states: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    start = tuple(range(n))
    q: deque[tuple[int, ...]] = deque([start])
    dist: dict[tuple[int, ...], int] = {start: 0}
    while q:
        state = q.popleft()
        d = dist[state]
        if d >= depth:
            continue
        for k in range(2, n + 1):
            nxt = tuple(reversed(state[:k])) + state[k:]
            if nxt in dist:
                continue
            dist[nxt] = d + 1
            q.append(nxt)
        if max_states and len(dist) >= max_states:
            break
    states = torch.tensor(list(dist.keys()), dtype=torch.long)
    distances = torch.tensor([dist[tuple(row.tolist())] for row in states], dtype=torch.float32)
    return states, distances

def random_walk_from_goal(
    n: int,
    *,
    batch_size: int,
    length: int,
    device: str | torch.device = "cpu",
    avoid_inverse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    states = goal_state(n, device=device).view(1, -1).repeat(batch_size, 1)
    depths = torch.zeros(batch_size, dtype=torch.float32, device=device)
    last = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    for step in range(1, length + 1):
        actions = torch.randint(2, n + 1, (batch_size,), dtype=torch.long, device=device)
        if avoid_inverse and n > 2:
            same = actions == last
            while same.any():
                actions[same] = torch.randint(2, n + 1, (int(same.sum()),), dtype=torch.long, device=device)
                same = actions == last
        states = apply_flip_actions(states, actions.view(-1, 1), normalize=False).squeeze(1)
        depths.fill_(float(step))
        last = actions
    return states, depths

def cayleypy_pancake_graph(n: int, device: str | torch.device | None = None) -> Any:
    graph = cp.CayleyGraph(cp.PermutationGroups.pancake(n))
    if device is not None and hasattr(graph, "to"):
        graph = graph.to(device)
    return graph
