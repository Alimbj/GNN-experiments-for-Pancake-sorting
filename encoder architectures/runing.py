if __name__ == "__main__":
    df_test = pd.read_csv('df_test.csv')
    beam_cfg = SearchConfig(
        beam_width=256, action_branching=44, max_steps_extra=10,
        device="cuda", amp=True
    )
    results = run_ablation_series(
        presets=["base", "+base", "+fourier"],
        n=12,
        iterations=1000,
        seed=42,
        eval_samples=10,
        df_test=df_test,
        beam_config=beam_cfg,
    )
    print("\n📊 Итоги абляций (оценка ОБОБЩЕНИЯ на реальном датасете):")
    print(f"{'Пресет':<12} {'TrainSR':>10} {'TestSR':>10} {'ΔTest':>10} {'Время':>10}")
    print("-" * 54)
    for r in results:
        train_sr = f"{r['train_sr']:.3f}" if not np.isnan(r["train_sr"]) else "—"
        test_sr = f"{r['test_sr']:.3f}" if r["test_sr"] is not None else "N/A"
        delta = compute_delta_sr(r["test_sr"], 
                                next((x["test_sr"] for x in results if x["preset"]=="base" and x["test_sr"] is not None), None))
        print(f"{r['preset']:<12} {train_sr:>10s} {test_sr:>10s} {delta:>10s} {r['train_time_min']:>9.1f}min")
    print("\nИнтерпретация:")
    print("  • TrainSR = success rate на синтетических данных (диагностика обучения)")
    print("  • TestSR = success rate на реальных перестановках (обобщающая способность) ← главная метрика")
    print("  • ΔTest = изменение TestSR относительно baseline (положительно = лучше обобщает)")
    print("  • Если ΔTest < +0.02 → компонент, вероятно, не улучшает обобщение")
