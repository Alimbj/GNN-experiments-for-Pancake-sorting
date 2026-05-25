
@dataclass(frozen=True)
class SoftBellmanConfig:
    gamma: float = 0.995
    tau: float = 0.35
    policy_tau: float = 0.45
    action_samples: int = 44
    state_chunk_size: int = 32
    value_batch_size: int = 256
    target_clip: float | None = None
    use_neighbor_value: bool = False
    target_mix_old: float = 0.15

def mellowmin(values: torch.Tensor, *, tau: float, dim: int = -1) -> torch.Tensor:
    if tau <= 0: return values.min(dim=dim).values
    n_actions = values.shape[dim]
    return -tau * (torch.logsumexp(-values / tau, dim=dim) - math.log(max(1, n_actions)))

@torch.inference_mode()
def evaluate_model_values_batched(model: nn.Module, states: torch.Tensor, *, batch_size: int = 512, amp: bool = True, use_neighbors: bool = True) -> torch.Tensor:
    device = next(model.parameters()).device
    x = as_batched_long(states, device=device)
    outputs: list[torch.Tensor] = []
    enabled = bool(amp and device.type == "cuda")
    for start in range(0, x.shape[0], batch_size):
        chunk = x[start : start + batch_size]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled):
            if hasattr(model, "heuristic"):
                value = model.heuristic(chunk, amp=False, use_neighbors=use_neighbors)
            else:
                value = model(chunk)[0]
            outputs.append(value.float().detach())
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, dtype=torch.float32, device=device)

@torch.no_grad()
def compute_bellman_batch(states: torch.Tensor, target_model: NeighborAttentionValuePolicyNet, *, bellman: SoftBellmanConfig, actions: torch.Tensor | None = None, amp: bool = True, policy_fraction: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(target_model.parameters()).device
    x_all = as_batched_long(states, device=device, n=target_model.config.n)
    n = x_all.shape[-1]
    out_targets, out_actions, out_policy = [], [], []
    for start in range(0, x_all.shape[0], max(1, bellman.state_chunk_size)):
        x = x_all[start : start + bellman.state_chunk_size]
        _, logits = target_model(x, use_neighbors=False)
        chunk_actions = None if actions is None else actions[start : start + x.shape[0]].to(device)
        neighbors, used_actions = transition_sampled_neighbors(x, num_actions=bellman.action_samples, actions=chunk_actions, policy_logits=logits, policy_fraction=policy_fraction)
        if used_actions.shape[1] == 0:
            targets = torch.zeros(x.shape[0], dtype=torch.float32, device=device)
            policy = torch.empty(x.shape[0], 0, dtype=torch.float32, device=device)
        else:
            flat = neighbors.reshape(x.shape[0] * used_actions.shape[1], n)
            values = evaluate_model_values_batched(target_model, flat, batch_size=bellman.value_batch_size, amp=amp, use_neighbors=bellman.use_neighbor_value).reshape(x.shape[0], used_actions.shape[1])
            q = 1.0 + bellman.gamma * values
            targets = mellowmin(q, tau=bellman.tau, dim=1)
            gap = gap_heuristic(x)
            margin = gap // 10  # ← адаптивный: 0 для gap<10, 1 для 10-19, и т.д.
            targets = targets.clamp(min=gap - margin)
            policy = torch.softmax(-q.detach() / max(1e-6, bellman.policy_tau), dim=1)
        if bellman.target_clip is not None: targets = targets.clamp(0.0, bellman.target_clip)
        targets[is_goal(x, n).to(device)] = 0.0
        out_targets.append(targets.detach()); out_actions.append(used_actions.detach()); out_policy.append(policy.detach())
    return torch.cat(out_targets), torch.cat(out_actions), torch.cat(out_policy)

compute_bellman_targets = lambda states, target_model, *, bellman=SoftBellmanConfig(), actions=None, amp=True, policy_fraction=0.5: compute_bellman_batch(states, target_model, bellman=bellman, actions=actions, amp=amp, policy_fraction=policy_fraction)[0]

@dataclass(frozen=True)
class SearchConfig:
    beam_width: int = 512
    action_branching: int = 44
    max_steps_extra: int = 25
    batch_size: int = 256
    policy_weight: float = 0.35
    gap_weight: float = 0.00
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

@dataclass(frozen=True)
class StreamingSearchResult:
    path_found: bool
    path_length: int
    expanded: int

@torch.inference_mode()
def streaming_neural_beam_search(start: torch.Tensor, model: NeighborAttentionValuePolicyNet, *, n: int, config: SearchConfig = SearchConfig()) -> StreamingSearchResult:
    device = torch.device(config.device)
    model = model.eval().to(device)
    start = as_batched_long(start, device=device, n=n)
    if is_goal(start, n).item(): return StreamingSearchResult(True, 0, 0)
    beam = start
    expanded = 0
    seen: set[tuple[int, ...]] = {tuple(int(v) for v in start[0].detach().cpu().tolist())}
    max_steps = n + config.max_steps_extra
    for depth in range(1, max_steps + 1):
        logits = model.policy_logits(beam, amp=config.amp, use_neighbors=False)
        top_actions = torch.topk(logits, k=min(config.action_branching, n - 1), dim=1).indices + 2
        strat = stratified_flip_actions(beam.shape[0], n, min(6, n - 1), device=device)
        actions = torch.cat([top_actions, strat], dim=1)
        children = apply_flip_actions(beam, actions, normalize=False).reshape(-1, n)
        expanded += int(children.shape[0])
        if is_goal(children, n).any().item(): return StreamingSearchResult(True, depth, expanded)
        keys = [tuple(int(v) for v in row.detach().cpu().tolist()) for row in children]
        keep_mask = torch.tensor([key not in seen for key in keys], dtype=torch.bool, device=device)
        children = children[keep_mask]
        keys = [key for key, keep in zip(keys, keep_mask.detach().cpu().tolist(), strict=True) if keep]
        if children.numel() == 0: return StreamingSearchResult(False, depth, expanded)
        for key in keys: seen.add(key)
        child_values = evaluate_model_values_batched(model, children, batch_size=config.batch_size, amp=config.amp, use_neighbors=False)
        child_logits = model.policy_logits(children, amp=config.amp, use_neighbors=False)
        probs = torch.softmax(child_logits, dim=-1)
        exploration_bonus = 1.5 * probs.max(dim=1).values / (1.0 + probs.max(dim=1).values)
        scores = child_values - exploration_bonus + config.gap_weight * gap_heuristic(children).to(device)
        keep = min(config.beam_width, children.shape[0])
        beam = children[torch.topk(scores, keep, largest=False).indices]  # ← largest=False, т.к. scores - это расстояние
    return StreamingSearchResult(False, max_steps, expanded)

@dataclass(frozen=True)
class EvalRecord:
    n: int
    heuristic: str
    path_found: bool
    path_length: int
    runtime_s: float

def summarize(records: list[EvalRecord]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for heuristic in sorted({r.heuristic for r in records}):
        group = [r for r in records if r.heuristic == heuristic]
        successes = [r for r in group if r.path_found]
        out[heuristic] = {"success_rate": len(successes) / max(1, len(group)), "mean_path_length": sum(r.path_length for r in successes) / max(1, len(successes)), "mean_runtime_s": sum(r.runtime_s for r in group) / max(1, len(group))}
    return out

@torch.inference_mode()
def training_diagnostics(model: NeighborAttentionValuePolicyNet, *, n: int, samples: int, search_config: SearchConfig, use_frontier: bool = True) -> dict[str, float]:
    """
    Исправленная диагностика: использует frontier curriculum + уменьшенный beam_width.
    """
    device = next(model.parameters()).device
    model = model.eval().to(device)
    
    # Генерируем тестовые состояния через frontier (если включено)
    test_states = []
    for _ in range(samples):
        if use_frontier and success_by_state:
            depth = sample_from_frontier(n, success_by_state, n)
            depth = max(2, min(depth, int(1.5 * n)))
        else:
            depth = random.randint(max(2, n // 3), n)
        state, _ = random_walk_from_goal(n, batch_size=1, length=depth, device=device)
        if not is_goal(state, n).item():
            test_states.append(state[0])
    
    if not test_states:
        return {"diag/goal_value": 0.0, "diag/beam_success_rate": 0.0, "diag/path_len": 0.0}
    
    # Уменьшенный beam_width для честной оценки
    eval_config = SearchConfig(
        beam_width=min(search_config.beam_width, 128),
        action_branching=search_config.action_branching,
        max_steps_extra=search_config.max_steps_extra,
        device=device,
        amp=search_config.amp,
        policy_weight=search_config.policy_weight,
        gap_weight=search_config.gap_weight,
    )
    
    beam = [streaming_neural_beam_search(start, model, n=n, config=eval_config) for start in test_states]
    beam_success = [r for r in beam if r.path_found]
    
    goal = goal_state(n, device=device).view(1, -1)
    goal_value = model.heuristic(goal, amp=search_config.amp, use_neighbors=False)[0]
    
    return {
        "diag/goal_value": float(goal_value.item()),
        "diag/beam_success_rate": len(beam_success) / max(1, len(beam)),
        "diag/path_len": sum(r.path_length for r in beam_success) / max(1, len(beam_success)) if beam_success else 0.0,
    }
