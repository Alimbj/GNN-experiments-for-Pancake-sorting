# Standalone Pancake Graph n=12 Experiments

This repository is organized for Kaggle, Jupyter, and Colab: **one Python file equals one complete experiment**. Each file in `experiments/` is self-contained and can be copied into a notebook cell or run directly with `python`.

## Design rules

- No YAML configs.
- No config registries.
- No dependency-injection framework.
- Hyperparameters are constants at the top of each experiment file.
- Every experiment contains the same Pancake environment, replay buffer, sampled-action training loop, streaming beam evaluator, and metrics protocol.
- No full Pancake graph materialization and no full frontier expansion.
- Training and search use sampled actions/neighbors only.
- The AVI target is always a Bellman target from the EMA model. The code never uses `target = gap`, `loss(pred, gap)`, or reward shaping from GAP.
- GAP is used as an explicit beam-search baseline and, only when a paper-mode constant is manually set to `oracle-assisted`, as an oracle feature source for methodology comparisons.
- The default learned experiments do not receive GAP counts, GAP deltas, heuristic gradients, GAP auxiliary losses, or GAP replay prioritization.

## Paper methodology modes for TDA

The TDA experiment exposes a top-level constant:

```python
TDA_MODE = "topology-aware"
```

Use it to make leakage boundaries explicit in a paper or benchmark table:

| mode | allowed information | intended interpretation |
| --- | --- | --- |
| `pure` | permutation geometry, transformer/action structure, local prefix-flip graph distances, Kendall tau metric | hardest mode: can the model discover useful heuristic structure without breakpoint/GAP channels? |
| `topology-aware` | everything in `pure`, plus breakpoint/local-adjacency topology inside the Vietoris-Rips metric | intrinsic graph-geometry learning; default and most scientifically reasonable mode |
| `oracle-assisted` | everything in `topology-aware`, plus GAP-derived metric components | engineering/distillation-style upper comparison; weaker scientifically |

The TDA feature vector itself is intentionally compact and persistence-only: Betti/persistence counts, persistence entropy, max persistence, and mean persistence for H0/H1. It does not append raw pairwise-distance summaries such as metric-space mean/std.

The TDA local metric no longer uses coordinate-mismatch distance. It uses Pancake-aware distances: Kendall tau and local prefix-flip graph distance in `pure`, breakpoint topology in `topology-aware`, and GAP components only in `oracle-assisted`.

## Graph metric contrastive pretraining

`experiments/contrastive_trajectory_pretraining.py` now samples positives from nearby true prefix-flip graph neighborhoods (`distance <= 2` by construction) and negatives from long random walks from the same anchors. This makes the objective graph-metric learning rather than random one-flip augmentation.

## Run one experiment

```bash
python experiments/baseline_sampled_attention.py
```

Every script prints:

- training `train_loss`
- learned-model beam `success_rate` and `average_solution_length`
- GAP baseline beam `success_rate` and `average_solution_length`
- optional random baseline metrics

The final parser-friendly line has this form:

```text
FINAL train_loss=... beam_success_rate=... average_solution_length=... gap_success_rate=... gap_average_solution_length=...
```

## Experiments

- `experiments/baseline_sampled_attention.py`
- `experiments/edge_conditioned_action_attention.py`
- `experiments/relative_flip_positional_encoding.py`
- `experiments/sparse_graph_transformer.py`
- `experiments/neural_algorithmic_reasoning.py`
- `experiments/contrastive_trajectory_pretraining.py`
- `experiments/cycle_aware_attention.py`
- `experiments/laplacian_positional_encoding.py`
- `experiments/tda_persistent_homology.py`
- `experiments/permutation_equivariant_transformer.py`

## Notebook workflow

Open any experiment file, edit constants such as `TRAIN_STEPS`, `BATCH_SIZE`, `BEAM_WIDTH`, `EVAL_EPISODES`, or `TDA_MODE`, then run the file. Defaults are intentionally small enough for debugging;  research-grade benchmark runs on RTX 4060 8GB or Kaggle T4.
