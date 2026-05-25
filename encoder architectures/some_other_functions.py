
def grad_scaler(device: str, amp: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=amp and str(device).startswith("cuda"))

def save_checkpoint(path: str | Path, model: NeighborAttentionValuePolicyNet, model_config: ModelConfig, *, train_config: StableAVIConfig | None = None, target_model: NeighborAttentionValuePolicyNet | None = None, replay: SourceWeightedReplayBuffer | None = None, iteration: int | None = None, kind: str = "stable-avi") -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": unwrap_compiled_model(model).state_dict(), "model_config": asdict(model_config), "kind": kind}
    if train_config is not None: payload["train_config"] = asdict(train_config)
    if target_model is not None: payload["target_model"] = unwrap_compiled_model(target_model).state_dict()
    if iteration is not None: payload["iteration"] = iteration
    if replay is not None:
        replay_path = path.with_suffix(".replay.pt"); replay.save(replay_path); payload["replay_path"] = str(replay_path)
    torch.save(payload, path)

def ema_update_target(online: nn.Module, target: nn.Module, *, tau: float) -> None:
    with torch.no_grad():
        for target_param, online_param in zip(target.parameters(), online.parameters(), strict=True): target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)
        for target_buffer, online_buffer in zip(target.buffers(), online.buffers(), strict=True): target_buffer.copy_(online_buffer)

def _add_anchor_batch(replay: SourceWeightedReplayBuffer, config: StableAVIConfig, *, progress: float = 0.0) -> None:
    goal = goal_state(config.n).view(1, -1).repeat(config.anchor_goal_repeat, 1)
    goal_targets = torch.zeros(config.anchor_goal_repeat, dtype=torch.float32)
    if config.anchor_shell_depth > 0:
        shell_states, shell_targets = exact_bfs_shells(config.n, depth=config.anchor_shell_depth)
        states, targets = torch.cat([goal, shell_states], dim=0), torch.cat([goal_targets, shell_targets], dim=0)
    else: states, targets = goal, goal_targets
    replay.add(states, targets, confidence=5.0, priorities=torch.full_like(targets, 5.0), source="anchor", replace_if_lower=True)

def _seed_replay(replay: SourceWeightedReplayBuffer, target_model: NeighborAttentionValuePolicyNet, config: StableAVIConfig) -> None:
    _add_anchor_batch(replay, config, progress=0.0)
    _add_walk_curriculum(replay, config, count=config.seed_random_walks, source="walk_curriculum", seed_phase=True, iteration=0)
    states, _, _, _ = replay.sample(min(config.batch_size, len(replay)), device=config.device)
    targets = compute_bellman_targets(states, target_model, bellman=config.bellman, amp=config.amp).cpu()
    replay.add(states.cpu(), targets, source="bellman_random", confidence=1.0, replace_if_lower=False)
