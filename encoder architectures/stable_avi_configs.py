
@dataclass(frozen=True)
class StableAVIConfig:
    n: int = 45
    iterations: int = 3000
    updates_per_iteration: int = 2
    batch_size: int = 1024
    train_microbatch_size: int = 256
    bellman_warmup_iterations: int = 300
    lr: float = 3e-4
    weight_decay: float = 1e-4
    replay_capacity: int = 500_000
    source_weights: dict[str, float] | None = None
    seed: int = 123
    target_update_interval: int = 10
    seed_random_walks: int = 2048
    seed_walk_length_min: int = 5
    seed_walk_length_max: int = 20
    random_walk_width: int = 256
    guided_batch_size: int = 256
    guided_walk_length: int = 35
    guided_action_samples: int = 44
    guided_eval_batch_size: int = 256
    anchor_shell_depth: int = 2
    anchor_goal_repeat: int = 1000
    bellman: SoftBellmanConfig = field(default_factory=SoftBellmanConfig)
    target_ema_tau: float = 0.01
    policy_loss_weight: float = 0.75
    eval_interval: int = 500
    eval_samples: int = 10
    checkpoint_interval: int = 500
    debug_memory: bool = False
    work_dir: str = "avi_outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    walk_decay_start: int = 50
    walk_decay_end: int = 10000
    model_guided_start: int = 1800

RLConfig = StableAVIConfig
