
class SourceWeightedReplayBuffer:
    def __init__(self, capacity: int = 500_000, *, n: int | None = None, source_weights: dict[str, float] | None = None, prioritized: bool = True, freshness_half_life: int | None = None, pin_memory: bool | None = None) -> None:
        self.capacity = int(capacity)
        self.n = n
        # 🔥 Добавлен новый источник "beam_failure" с высоким весом
        self.source_weights = source_weights or {
            "anchor": 0.5, "walk_curriculum": 3.0, "bellman_guided": 1.5, 
            "bellman_random": 2.0, "search": 4.0, "gap_ascent_curriculum": 2.5, 
            "model_guided": 1.0, "beam_failure": 8.0  # ← новый источник
        }
        self._source_to_id = {name: i for i, name in enumerate(self.source_weights)}
        self._id_to_source = {i: name for name, i in self._source_to_id.items()}
        self.prioritized = prioritized
        self.freshness_half_life = freshness_half_life
        self.pin_memory = bool(torch.cuda.is_available() if pin_memory is None else pin_memory)
        self._states: torch.Tensor | None = None
        self._targets: torch.Tensor | None = None
        self._confidence: torch.Tensor | None = None
        self._priorities: torch.Tensor | None = None
        self._source_ids: torch.Tensor | None = None
        self._steps: torch.Tensor | None = None
        self._keys: list[tuple[int, ...] | None] = [None] * self.capacity
        self._index: dict[tuple[int, ...], int] = {}
        self._size = 0
        self._write = 0
        self._step = 0
        self._weight_cache: torch.Tensor | None = None
        self._weight_cache_step = -1
        self.anchor_source_id = self._source_to_id.get("anchor", -1)
    def __len__(self) -> int: return self._size
    def _allocate(self, n: int) -> None:
        if self._states is not None: return
        self.n = int(n)
        try:
            self._states = torch.empty((self.capacity, self.n), dtype=torch.long, pin_memory=self.pin_memory)
            self._targets = torch.empty(self.capacity, dtype=torch.float32, pin_memory=self.pin_memory)
            self._confidence = torch.empty(self.capacity, dtype=torch.float32, pin_memory=self.pin_memory)
            self._priorities = torch.empty(self.capacity, dtype=torch.float32, pin_memory=self.pin_memory)
            self._source_ids = torch.empty(self.capacity, dtype=torch.long, pin_memory=self.pin_memory)
            self._steps = torch.empty(self.capacity, dtype=torch.long, pin_memory=self.pin_memory)
        except RuntimeError:
            self.pin_memory = False
            self._states = torch.empty((self.capacity, self.n), dtype=torch.long)
            self._targets = torch.empty(self.capacity, dtype=torch.float32)
            self._confidence = torch.empty(self.capacity, dtype=torch.float32)
            self._priorities = torch.empty(self.capacity, dtype=torch.float32)
            self._source_ids = torch.empty(self.capacity, dtype=torch.long)
            self._steps = torch.empty(self.capacity, dtype=torch.long)
    def _source_id(self, source: str) -> int:
        if source not in self._source_to_id:
            idx = len(self._source_to_id)
            self._source_to_id[source] = idx
            self._id_to_source[idx] = source
            self.source_weights.setdefault(source, 1.0)
        return self._source_to_id[source]
    def _source_weight_tensor(self) -> torch.Tensor:
        max_id = max(self._id_to_source.keys(), default=0)
        weights = torch.ones(max_id + 1, dtype=torch.float32)
        for idx, source in self._id_to_source.items():
            weights[idx] = float(self.source_weights.get(source, 1.0))
        return weights
    def _invalidate(self) -> None:
        self._weight_cache = None
        self._weight_cache_step = -1
    def add(self, states: torch.Tensor, targets: torch.Tensor, *, confidence: torch.Tensor | float = 1.0, priorities: torch.Tensor | None = None, source: str = "bellman_guided", replace_if_lower: bool = True) -> None:
        x = as_batched_long(states).detach().cpu()
        if x.numel() == 0: return
        self._allocate(x.shape[1])
        assert self._states is not None and self._targets is not None and self._confidence is not None
        assert self._priorities is not None and self._source_ids is not None and self._steps is not None
        y = torch.as_tensor(targets, dtype=torch.float32).flatten().detach().cpu()
        c = confidence.float().flatten().detach().cpu() if isinstance(confidence, torch.Tensor) else torch.full_like(y, float(confidence))
        p = torch.ones_like(y) if priorities is None else priorities.float().flatten().detach().cpu().clamp_min(1e-6)
        source_id = self._source_id(source)
        source_weight = self.source_weights.get(source, 1.0)
        new_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        new_indices, update_indices = [], []
        for i, state_tensor in enumerate(x):
            key = tuple(int(v) for v in state_tensor.tolist())
            if key not in self._index:
                new_mask[i] = True
                new_indices.append((i, key))
            else:
                update_indices.append((i, key, self._index[key]))
        if new_indices:
            new_idxs = [i for i, _ in new_indices]
            new_keys = [k for _, k in new_indices]
            write_positions = [(self._write + j) % self.capacity for j in range(len(new_idxs))]
            self._states[write_positions] = x[new_idxs]
            self._targets[write_positions] = y[new_idxs]
            self._confidence[write_positions] = c[new_idxs]
            self._priorities[write_positions] = p[new_idxs]
            self._source_ids[write_positions] = source_id
            self._steps[write_positions] = self._step + torch.arange(len(new_idxs), dtype=torch.long)
            for j, key in enumerate(new_keys):
                pos = write_positions[j]
                self._keys[pos] = key
                self._index[key] = pos
            self._write = (self._write + len(new_idxs)) % self.capacity
            self._size = min(self._size + len(new_idxs), self.capacity)
            self._step += len(new_idxs)
        for i, key, old_idx in update_indices:
            old_target = float(self._targets[old_idx])
            if replace_if_lower and float(y[i]) < old_target:
                mixed = float(y[i])
            else:
                old_source = self._id_to_source.get(int(self._source_ids[old_idx]), "bellman_guided")
                old_mass = float(self._confidence[old_idx]) * self.source_weights.get(old_source, 1.0)
                new_mass = float(c[i]) * source_weight
                mixed = (old_target * old_mass + float(y[i]) * new_mass) / max(1e-6, old_mass + new_mass)
            self._priorities[old_idx] = max(float(p[i]), abs(mixed - old_target) + 1e-6)
            self._targets[old_idx] = float(mixed)
            self._confidence[old_idx] = float(self._confidence[old_idx]) + float(c[i])
            self._source_ids[old_idx] = source_id if source_weight >= 1.0 else self._source_ids[old_idx]
            self._steps[old_idx] = self._step
        self._invalidate()
    def _sampling_weights(self) -> torch.Tensor:
        if self._weight_cache is not None and self._weight_cache_step == self._step:
            return self._weight_cache
        assert self._confidence is not None and self._priorities is not None and self._source_ids is not None and self._steps is not None
        source_weights = self._source_weight_tensor()[self._source_ids[: self._size]]
        weights = source_weights * self._priorities[: self._size] * self._confidence[: self._size].clamp_min(1.0).pow(0.25)
        if self.freshness_half_life is not None:
            age = (self._step - self._steps[: self._size]).clamp_min(0).float()
            weights = weights * torch.pow(torch.tensor(0.5), age / max(1, self.freshness_half_life))
        self._weight_cache = weights.clamp_min(1e-6)
        self._weight_cache_step = self._step
        return self._weight_cache
    def sample(self, batch_size: int, *, device: str | torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        assert self._states is not None and self._targets is not None and self._confidence is not None and self._source_ids is not None
        k = min(batch_size, self._size)
        if self.prioritized:
            idx = torch.multinomial(self._sampling_weights(), k, replacement=self._size < batch_size)
        else:
            idx = torch.randint(0, self._size, (k,)) if self._size < batch_size else torch.randperm(self._size)[:k]
        states = self._states.index_select(0, idx).to(device, non_blocking=True)
        targets = self._targets.index_select(0, idx).to(device, non_blocking=True)
        sampled_source_ids = self._source_ids.index_select(0, idx).to(device, non_blocking=True)
        weights = (self._source_weight_tensor()[sampled_source_ids.cpu()] * self._confidence.index_select(0, idx).clamp_min(1.0).pow(0.5)).to(device, non_blocking=True)
        return states, targets, weights, sampled_source_ids
    def boost_priorities(self, states: torch.Tensor, priorities: torch.Tensor, *, blend: float = 0.5) -> None:
        if self._size == 0 or self._priorities is None: return
        x = as_batched_long(states).detach().cpu()
        p = torch.as_tensor(priorities, dtype=torch.float32).flatten().detach().cpu().clamp_min(1e-6)
        indices_to_update = []
        for state_tensor, priority in zip(x, p, strict=True):
            idx = self._index.get(tuple(int(v) for v in state_tensor.tolist()))
            if idx is not None:
                indices_to_update.append((idx, float(priority)))
        if indices_to_update:
            idx_tensor = torch.tensor([i for i, _ in indices_to_update], dtype=torch.long)
            priority_tensor = torch.tensor([p for _, p in indices_to_update], dtype=torch.float32)
            old_priorities = self._priorities[idx_tensor]
            new_priorities = torch.maximum(old_priorities, (1.0 - blend) * old_priorities + blend * priority_tensor)
            self._priorities[idx_tensor] = new_priorities
            self._invalidate()
    def source_counts(self) -> dict[str, int]:
        if self._size == 0 or self._source_ids is None: return {}
        ids, counts = torch.unique(self._source_ids[: self._size], return_counts=True)
        return {self._id_to_source.get(int(i), "unknown"): int(c) for i, c in zip(ids, counts, strict=True)}
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capacity": self.capacity, "n": self.n, "source_weights": self.source_weights, "prioritized": self.prioritized, "freshness_half_life": self.freshness_half_life, "step": self._step, "size": self._size, "write": self._write, "states": None if self._states is None else self._states[: self._size].clone(), "targets": None if self._targets is None else self._targets[: self._size].clone(), "confidence": None if self._confidence is None else self._confidence[: self._size].clone(), "priorities": None if self._priorities is None else self._priorities[: self._size].clone(), "source_ids": None if self._source_ids is None else self._source_ids[: self._size].clone(), "steps": None if self._steps is None else self._steps[: self._size].clone(), "source_to_id": self._source_to_id}, path)

ReplayBuffer = SourceWeightedReplayBuffer
