# Многомерный frontier: ключ = (depth, gap_bin, td_error_bin)
success_by_state: dict[tuple[int, int, int], float] = {}  # (depth, gap_bin, error_bin) -> EMA success
hard_buffer: list[torch.Tensor] = []
hard_priorities: list[float] = []
hard_targets: list[torch.Tensor] = []
_curriculum_depths: list[int] = []

# Вспомогательная функция для биннинга
def _bin_value(val: float, bins: int, min_val: float, max_val: float) -> int:
    """Квантует значение в [0, bins-1]."""
    normalized = (val - min_val) / max(1e-6, max_val - min_val)
    return min(bins - 1, max(0, int(normalized * bins)))

# Обновление frontier только на hard states с многомерным ключом
def _update_frontier(depth: int, gap: float, td_error: float, solved: bool, alpha: float = 0.05) -> None:
    """Обновляет success_by_state только если состояние действительно сложное."""
    # Бинним признаки для создания ключа
    gap_bin = _bin_value(gap, bins=5, min_val=0, max_val=12)  # gap ∈ [0, n]
    error_bin = _bin_value(td_error, bins=5, min_val=0, max_val=5)  # td_error ∈ [0, 5+]
    
    key = (depth, gap_bin, error_bin)
    
    # Только hard states обновляют frontier
    is_hard = (td_error > 0.5) or (gap > 6) or (solved == False and depth > 8)
    if is_hard:
        if key not in success_by_state:
            success_by_state[key] = float(solved)
        else:
            success_by_state[key] = (1 - alpha) * success_by_state[key] + alpha * float(solved)

# Сэмплирование из многомерного фронта
def sample_from_frontier(n: int, success_by_state: dict, config_n: int) -> int:
    """Сэмплирует глубину из многомерного фронта обучения."""
    max_depth = int(1.5 * config_n)
    
    # Находим ключи в целевом диапазоне успеха
    frontier_keys = [k for k, s in success_by_state.items() if 0.3 <= s <= 0.8]
    
    if frontier_keys:
        # Выбираем случайный ключ из фронта и возвращаем его глубину
        chosen_key = random.choice(frontier_keys)
        return int(chosen_key[0])
    
    # Fallback: гауссиана вокруг средней глубины известных состояний
    known_depths = [k[0] for k in success_by_state.keys()]
    if known_depths:
        center = np.mean(known_depths)
        sigma = max(2, center * 0.2)
        for _ in range(50):
            d = round(random.gauss(center, sigma))
            if 2 <= d <= max_depth:
                return int(d)
    
    return int(config_n // 2)  # последний fallback
