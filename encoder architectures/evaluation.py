
@torch.inference_mode()
def evaluate_success_rate(
    model: NeighborAttentionValuePolicyNet,
    n: int,
    num_samples: int = 50,
    max_walk_length: int = 20,
    beam_config: SearchConfig | None = None,
    device: str | torch.device = None,
) -> dict[str, float]:
    if beam_config is None:
        beam_config = SearchConfig(
            beam_width=256, action_branching=44, max_steps_extra=30,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            amp=True,
        )
    device = torch.device(beam_config.device)
    model = model.eval().to(device)
    test_data = []
    for _ in range(num_samples):
        length = random.randint(5, max_walk_length)
        state, _ = random_walk_from_goal(n, batch_size=1, length=length, device=device)
        if not is_goal(state, n).item():
            test_data.append((state[0], length))
    if not test_data:
        return {"success_rate": 1.0, "avg_path_length": 0.0, "avg_expanded": 0.0, "samples": 0, "avg_scramble_length": 0.0}
    results = []
    scramble_lengths = []
    for state, scramble_len in test_data:
        result = streaming_neural_beam_search(state, model, n=n, config=beam_config)
        results.append(result)
        scramble_lengths.append(scramble_len)
    solved = [r for r in results if r.path_found]
    success_rate = len(solved) / len(results)
    avg_path_length = sum(r.path_length for r in solved) / len(solved) if solved else float("inf")
    avg_expanded = sum(r.expanded for r in results) / len(results)
    avg_scramble_length = sum(scramble_lengths) / len(scramble_lengths)
    scramble_by_length = {}
    for (state, slen), res in zip(test_data, results):
        if slen not in scramble_by_length:
            scramble_by_length[slen] = {"total": 0, "solved": 0}
        scramble_by_length[slen]["total"] += 1
        if res.path_found:
            scramble_by_length[slen]["solved"] += 1
    return {
        "success_rate": success_rate,
        "avg_path_length": avg_path_length,
        "avg_expanded": avg_expanded,
        "samples": len(results),
        "avg_scramble_length": avg_scramble_length,
        "scramble_by_length": scramble_by_length,
    }

def parse_permutation_cell(cell: Any) -> list[int]:
    if isinstance(cell, (list, tuple)):
        return [int(x) for x in cell]
    return [int(x) for x in str(cell).replace(",", " ").split()]

def evaluate_on_test_dataset(
    model: NeighborAttentionValuePolicyNet,
    df_test: pd.DataFrame,
    n: int,
    beam_config: SearchConfig,
    num_samples: int = 20,
) -> dict[str, float]:
    device = torch.device(beam_config.device)
    model = model.eval().to(device)
    df_n = df_test[df_test["n"] == n].head(num_samples)
    test_states = [
        torch.tensor(parse_permutation_cell(row["permutation"]), dtype=torch.long, device=device).unsqueeze(0)
        for _, row in df_n.iterrows()
    ]
    if not test_states:
        return {"success_rate": 0.0, "avg_path_length": 0.0, "avg_expanded": 0.0, "samples": 0}
    results = [
        streaming_neural_beam_search(state, model, n=n, config=beam_config)
        for state in tqdm(test_states, desc=f"Eval n={n}", leave=False)
    ]
    solved = [r for r in results if r.path_found]
    return {
        "success_rate": len(solved) / len(results),
        "avg_path_length": sum(r.path_length for r in solved) / len(solved) if solved else float("inf"),
        "avg_expanded": sum(r.expanded for r in results) / len(results),
        "samples": len(results),
    }

def compute_delta_sr(current_sr: float, baseline_sr: float | None) -> str:
    if baseline_sr is None:
        return "— (baseline)"
    delta = current_sr - baseline_sr
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"
