def default_model_config(n: int = 45) -> ModelConfig:
    return ModelConfig(n=n, d_model=312, depth=2, max_neighbors=44, neighbor_mode="sampled", use_policy_head=True)

def default_avi_config(n: int = 45) -> StableAVIConfig:
    return StableAVIConfig(n=n, iterations=5000, updates_per_iteration=2, batch_size=1024, train_microbatch_size=256, replay_capacity=500_000, seed_random_walks=2048, guided_batch_size=256, guided_walk_length=max(20, n), guided_action_samples=44, eval_interval=500, eval_samples=15, checkpoint_interval=500, device="cuda" if torch.cuda.is_available() else "cpu", amp=True)

def log_fn(step: int, metrics: dict[str, float]) -> None:
    if step != 1 and step % 50 != 0: return
    msg = f"[{step}] loss={metrics.get('loss', float('nan')):.4f} value={metrics.get('value_loss', float('nan')):.4f} policy={metrics.get('policy_loss', float('nan')):.4f} replay={int(metrics.get('replay_size', 0))} rw={metrics.get('curriculum/random_walk_steps_min', 0):.0f}-{metrics.get('curriculum/random_walk_steps_max', 0):.0f} ga={metrics.get('curriculum/gap_ascent_steps_actual', 0):.0f} mg={metrics.get('curriculum/model_guided_steps', 0):.0f} pf={metrics.get('curriculum/policy_fraction', 0):.2f}"
    for key in ["diag/beam_success_rate", "diag/path_len"]:
        if key in metrics: msg += f" | {key}={metrics[key]:.3f}"
    tqdm.write(msg)

# ============================================================================
# Ablation Protocol
# ============================================================================

def run_ablation_test(
    preset: str,
    n: int = 12,
    iterations: int = 200,
    seed: int = 42,
    work_dir: str = "ablation_test",
    eval_samples: int = 20,
    df_test: pd.DataFrame | None = None,
    beam_config: SearchConfig | None = None,
) -> dict:
    set_seed(seed)
    model_config = get_config_for_preset(preset, n=n)
    train_config = StableAVIConfig(
        n=n, iterations=iterations, batch_size=256,
        work_dir=f"{work_dir}/{preset}",
        eval_interval=50, checkpoint_interval=200,
        amp=True, device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if beam_config is None:
        beam_config = SearchConfig(
            beam_width=256, action_branching=44, max_steps_extra=30,
            device=train_config.device, amp=True
        )
    train_metrics_log = []
    def log_callback(step: int, metrics: dict):
        if step % 50 == 0:
            train_metrics_log.append(metrics)
            sr_train = metrics.get("diag/beam_success_rate", float("nan"))
            loss = metrics.get("loss", float("nan"))
            print(f"  [{step}] TrainSR={sr_train:.3f}, Loss={loss:.3f}")
    print(f"🔬 Запуск: {preset:10s} (n={n}, iters={iterations})")
    start_time = time.time()
    model = build_model(model_config, device=train_config.device)
    trained = train_stable_avi(model, model_config, train_config, log_callback=log_callback)
    train_time = time.time() - start_time
    test_sr, test_path_len, test_expanded = None, None, None
    if df_test is not None and not df_test[df_test["n"] == n].empty:
        test_result = evaluate_on_test_dataset(
            trained, df_test, n=n, num_samples=eval_samples,
            beam_config=beam_config
        )
        test_sr = test_result["success_rate"]
        test_path_len = test_result["avg_path_length"]
        test_expanded = test_result["avg_expanded"]
        print(f"✅ Test SR (df_test): {test_sr:.3f}, PathLen: {test_path_len:.1f}")
    final_train_metrics = train_metrics_log[-1] if train_metrics_log else {}
    train_sr = final_train_metrics.get("diag/beam_success_rate", float("nan"))
    return {
        "preset": preset,
        "train_sr": train_sr,
        "test_sr": test_sr,
        "test_path_len": test_path_len,
        "test_expanded": test_expanded,
        "train_time_min": train_time / 60,
        "samples_evaluated": eval_samples if df_test is not None else 0,
        "model": trained,
        "config": model_config,
    }

def run_ablation_series(
    presets: list[str] | None = None,
    n: int = 12,
    iterations: int = 1000,
    seed: int = 42,
    eval_samples: int = 10,
    df_test: pd.DataFrame | None = None,
    beam_config: SearchConfig | None = None,
) -> list[dict]:
    if presets is None:
        presets = ABLATION_PRESETS[:4]
    results = []
    baseline_test_sr = None
    print(f"📋 План: {presets}")
    print(f"📦 n={n}, iterations={iterations}, seed={seed}")
    if df_test is not None:
        print(f"🗄️  Оценка обобщения на реальных данных (df_test)")
    print()
    for preset in presets:
        res = run_ablation_test(
            preset, n=n, iterations=iterations, seed=seed,
            eval_samples=eval_samples, df_test=df_test, beam_config=beam_config
        )
        delta_test_sr = compute_delta_sr(res["test_sr"], baseline_test_sr)
        train_sr_str = f"{res['train_sr']:.3f}" if not np.isnan(res["train_sr"]) else "—"
        test_sr_str = f"{res['test_sr']:.3f}" if res["test_sr"] is not None else "N/A"
        print(f"✅ {preset:10s}: "
              f"TrainSR={train_sr_str:>8s} | "
              f"TestSR={test_sr_str:>8s} Δ={delta_test_sr:>8s} | "
              f"Time={res['train_time_min']:.1f}min\n")
        if preset == "base" and res["test_sr"] is not None:
            baseline_test_sr = res["test_sr"]
        results.append(res)
    return results
