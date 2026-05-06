# CNN Depth-Width Ratio Study

This repository contains the experiment code for the COMP5329/COMP4329 Assignment 2 project:

**An Empirical Investigation of Optimal Depth-to-Width Ratios in CNNs Under Fixed Parameter Budgets**

The code is designed for controlled empirical analysis rather than a single best-score run. It supports plain CNNs, residual CNNs, multiple width schedules, automatic parameter-budget matching, CIFAR-10/CIFAR-100, and paper-ready plots/tables.

## Setup

```bash
pip install -r requirements.txt
```

The code expects datasets under `data/` and does not download automatically unless `--download` is provided.

```text
data/
  cifar-10-batches-py/
  cifar-100-python/
```

## Train One Model

Plain CNN:

```bash
python -m src.train \
  --dataset cifar10 \
  --model-family plain \
  --width-schedule stagewise \
  --depth 4 \
  --width 64 \
  --epochs 100 \
  --seed 0 \
  --measure-flops \
  --track-grad-norm
```

Residual CNN:

```bash
python -m src.train \
  --dataset cifar10 \
  --model-family residual \
  --width-schedule stagewise \
  --depth 6 \
  --width 32 \
  --epochs 100 \
  --seed 0 \
  --measure-flops
```

Outputs are written to `results/`:

- `*_history.csv`: per-epoch loss, accuracy, throughput, gradient norm
- `*_summary.json`: final run-level summary

## Run A Manual Sweep

```bash
python -m src.sweep \
  --config configs/cifar10_budget_sweep.json \
  --epochs 100 \
  --seeds 0,1,2 \
  --measure-flops \
  --track-grad-norm
```

## Run A Fixed-Parameter-Budget Sweep

This is the recommended protocol for the paper. It searches for the width closest to a target parameter budget for each depth.

```bash
python -m src.sweep \
  --dataset cifar10 \
  --model-family plain \
  --width-schedule stagewise \
  --target-params 250000 \
  --depths 2,4,6,8,10,12 \
  --epochs 100 \
  --seeds 0,1,2 \
  --measure-flops \
  --track-grad-norm
```

For the residual comparison:

```bash
python -m src.sweep \
  --dataset cifar10 \
  --model-family residual \
  --width-schedule stagewise \
  --target-params 250000 \
  --depths 2,4,6,8,10,12 \
  --epochs 100 \
  --seeds 0,1,2 \
  --measure-flops
```

## Generate Figures And Tables

```bash
python -m src.plot_results --summary results/sweep_summary.csv
python -m src.analysis --summary results/sweep_summary.csv
```

Figures are written to `results/figures/`:

- `accuracy_vs_ratio.png`
- `generalization_gap_vs_ratio.png`
- `training_time_vs_ratio.png`
- `budget_error_vs_ratio.png`
- `accuracy_vs_flops.png`
- `test_accuracy_curves.png`

Tables are written to `results/tables/`:

- `model_ranking.md`
- `family_summary.md`

## Experiment Design Notes

Useful factors for the paper:

- **Depth-width ratio:** compare depth while matching parameter count as closely as possible.
- **Optimization stability:** use gradient norm and convergence curves to distinguish representation limits from training failure.
- **Generalization:** report train-test gap, not only best test accuracy.
- **Compute efficiency:** report FLOPs and wall-clock training time alongside accuracy.
- **Skip connections:** compare `plain` vs `residual` to test whether deep narrow models fail because of optimization.
- **Task difficulty:** use CIFAR-10 first; add CIFAR-100 if time allows to support claims about task complexity.

Recommended final runs:

1. CIFAR-10, plain CNN, 2-3 parameter budgets, 3 seeds.
2. CIFAR-10, residual CNN, same budgets, 3 seeds.
3. Optional CIFAR-100 subset or full CIFAR-100 for task-complexity analysis.
4. Report mean and standard deviation, plus FLOPs and parameter-budget error.

Avoid claiming a universal optimal ratio unless the result is consistent across datasets, budgets, and model families.
