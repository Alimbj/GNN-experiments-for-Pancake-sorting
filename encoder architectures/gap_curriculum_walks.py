# === Лимиты блужданий ===
RANDOM_WALK_MAX_DEPTH = lambda n: n + n // 2   # для random walks
GAP_WALK_MAX_DEPTH = lambda n: n + n // 10     # для gap-guided walks 

def _add_walk_curriculum(
    replay: SourceWeightedReplayBuffer,
    config: Any,
    *,
    count: int,
    source: str = "walk_curriculum",
    progress: float = 0.0,
    seed_phase: bool = False,
    iteration: int = 0,
    gap_batch_size: int | None = None,
) -> tuple[int, int]:
    """
    Генерирует случайные блуждания с frontier curriculum.
    Векторизованная генерация переменной длины + защищённые приоритеты.
    """
    effective_count = gap_batch_size if gap_batch_size is not None else int(count * (1.0 - min(1.0, iteration / config.walk_decay_end)))
    if effective_count <= 0:
        return (0, 0)
    _walk_cache: dict[tuple[int, int], torch.Tensor] = {}
    remaining = effective_count
    if seed_phase:
        min_d, max_d = config.seed_walk_length_min, config.seed_walk_length_max
    else:
        min_d, max_d = 2, RANDOM_WALK_MAX_DEPTH(config.n)
    
    # Оцениваем центр фронта для приоритетов
    # Вычисляем среднюю глубину из ключей в целевом диапазоне
    frontier_keys = [k for k, s in success_by_state.items() if 0.3 <= s <= 0.8]
    if frontier_keys:
        frontier_center = np.mean([k[0] for k in frontier_keys])  # k[0] = depth
    else:
        frontier_center = config.n // 2
        
        while remaining > 0:
            width = min(512, remaining)
        
        # Per-state depth sampling (здоровое распределение)
        depths_list = [sample_from_frontier(config.n, success_by_state, config.n) for _ in range(width)]
        depths_list = [max(min_d, min(d, max_d)) for d in depths_list]
        depths_tensor = torch.tensor(depths_list, dtype=torch.long)
        
        # Векторизованная генерация блужданий переменной длины
        # Генерируем все блуждания за один проход через batched random_walk
        states_list = []
        for d in set(depths_list):  # группируем по уникальным длинам
            mask = depths_tensor == d
            count_d = int(mask.sum().item())
            if count_d > 0:
                s, _ = random_walk_from_goal(config.n, batch_size=count_d, length=int(d), device="cpu")
                states_list.append((mask, s))
        
        # Собираем в правильном порядке
        states = torch.empty((width, config.n), dtype=torch.long)
        for mask, s in states_list:
            states[mask] = s
        
        # Приоритеты: дальше от фронта = важнее
        priorities = torch.tensor([
            1.0 + abs(d - frontier_center) / max(1, frontier_center)
            for d in depths_list
        ], dtype=torch.float32)
        
        # Логирование реальных глубин
        global _curriculum_depths
        _curriculum_depths.extend(depths_list)
        
        # replace_if_lower=True для curriculum samples (не дублировать лёгкие)
        replay.add(states, depths_tensor.float(), confidence=2.0, priorities=priorities, 
                  source="walk_curriculum", replace_if_lower=True)  # ✅ FIX
        remaining -= width
    
    return (min_d, max_d)

# ============================================================================
# 📈 Bellman, Training
# ============================================================================

@dataclass(frozen=True)
class GapAscentConfig:
    start_iter: int = 1000
    rampup_iters: int = 3500
    action_samples: int = 44
    batch_size: int = 512
    source_name: str = "gap_ascent_curriculum"
    priority: float = 2.0
    confidence: float = 1.0
    min_gap_threshold: float = 0.5
    temperature: float = 1.5

@torch.no_grad()
def _generate_gap_ascent_states(n: int, batch_size: int, steps: int, temperature: float, action_samples: int, min_gap: float, device: torch.device) -> torch.Tensor:
    states = goal_state(n, device=device).view(1, -1).expand(batch_size, -1)
    flips = torch.arange(2, n + 1, device=device)
    penalty = -20.0
    escape_prob = 0.05
    for _ in range(steps):
        m = min(action_samples, n - 1)
        idx = torch.randperm(flips.size(0), device=device)[:m]
        actions = flips[idx]
        neighbors = apply_flip_actions(states, actions, normalize=False)
        gaps = gap_heuristic(neighbors.reshape(-1, n)).reshape(batch_size, m)
        if min_gap > 0:
            gaps = torch.where(gaps < min_gap, torch.full_like(gaps, penalty), gaps)
        escape_mask = torch.rand(batch_size, m, device=device) < escape_prob
        gaps_escape = torch.where(escape_mask, torch.zeros_like(gaps), gaps)
        temperature = max(temperature, 1e-3)
        probs = torch.softmax(gaps_escape / temperature, dim=1)
        probs = (probs + 1e-12); probs = probs / probs.sum(dim=1, keepdim=True)
        chosen = torch.multinomial(probs, 1).squeeze(1)
        states = neighbors[torch.arange(batch_size, device=device), chosen]
    return states

def _inject_gap_curriculum(replay, config: GapAscentConfig, iteration: int, n: int, device: torch.device, dummy_target_base: float = 5.0) -> tuple[int, int]:
    if iteration < config.start_iter: return (0, 0, 0)
    max_steps = GAP_WALK_MAX_DEPTH(n)
    steps = random.randint(5, max_steps)
    states = _generate_gap_ascent_states(n=n, batch_size=config.batch_size, steps=steps, temperature=config.temperature, action_samples=config.action_samples, min_gap=config.min_gap_threshold, device=device)
    dummy_targets = torch.full((config.batch_size,), dummy_target_base, dtype=torch.float32, device=device)
    replay.add(states.cpu(), dummy_targets.cpu(), confidence=config.confidence, priorities=torch.full((config.batch_size,), config.priority, dtype=torch.float32), source=config.source_name, replace_if_lower=False)
    return (5, max_steps, steps)
