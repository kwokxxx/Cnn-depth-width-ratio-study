# CNN Depth-Width Ratio Study

This repository contains the code and Colab workflow for:

**An Empirical Investigation of Optimal Depth-to-Width Ratios in CNNs Under Fixed Parameter Budgets**

The project studies how CNN depth and width allocation affects CIFAR classification under controlled parameter budgets. The code supports plain CNNs, residual CNNs, fixed-budget width search, effective depth-width ratio analysis, FLOPs, gradient norms, plots, and paper-ready tables.

## Recommended Workflow: Colab

Use [experiment.ipynb](experiment.ipynb) as the main entry point.

In Colab:

1. Open `experiment.ipynb`.
2. Select **Runtime > Change runtime type > GPU**.
3. Run the notebook cells from top to bottom.
4. When prompted, mount Google Drive.

The notebook stores all project-related content in:

```text
MyDrive/5329A2/
```

Expected Drive layout:

```text
MyDrive/5329A2/project/   # cloned GitHub repository
MyDrive/5329A2/data/      # CIFAR-10 and optional CIFAR-100
MyDrive/5329A2/results/   # training logs, summaries, figures, tables
```

This means Colab runtime resets will not delete the dataset or experiment outputs.

## Main Experiments

The notebook first runs a smoke test, then two CIFAR-10 sweeps:

- **Plain CNN sweep:** main fixed-budget depth-width experiment.
- **Residual CNN sweep:** control experiment for deep-network optimization difficulty.

Default experiment settings in the notebook:

```python
TARGET_PARAMS = 250_000
DEPTHS = "2,4,6,8,10,12"
SEEDS = "0,1"
EPOCHS = 30
BATCH_SIZE = 128
```

For stronger final results, increase `EPOCHS` to `50` or `100` if GPU time allows.

With 6 depths, 2 seeds, and 100 epochs:

```text
6 * 2 * 100 = 1200 epochs per sweep
```

Running both plain and residual sweeps doubles this to 2400 total epochs.

## Output Files

Each sweep writes outputs under `MyDrive/5329A2/results/...`.

Important files:

- `sweep_summary.csv`: run-level results across depths and seeds.
- `*_history.csv`: per-epoch loss, accuracy, throughput, gradient norm.
- `*_summary.json`: summary for one training run.
- `figures/accuracy_vs_ratio.png`: accuracy vs effective depth-width ratio.
- `figures/generalization_gap_vs_ratio.png`: train-test gap vs effective ratio.
- `figures/accuracy_vs_flops.png`: accuracy vs compute cost.
- `tables/model_ranking.md`: paper-ready model ranking table.
- `tables/family_summary.md`: aggregate model-family summary.

## Architecture Ratio Definition

For stagewise CNNs, the base width is only the first-stage channel count. The code therefore records two ratios:

```text
base_depth_width_ratio = depth / base_width
effective_depth_width_ratio = depth / effective_width
```

where:

```text
effective_width = geometric mean of per-block channel counts
```

Figures and tables use `effective_depth_width_ratio` as the main x-axis. This avoids treating the base width as the true width of a stagewise network.

## Downsampling Policy

Spatial downsampling is capped at three reductions:

```text
32x32 -> 16x16 -> 8x8 -> 4x4
```

This prevents deeper models such as depth 10 or 12 from repeatedly collapsing the CIFAR feature maps to 1x1, which would confound depth effects with spatial-resolution loss.

## Local Usage

Colab is recommended, but the same commands work locally after installing dependencies:

```bash
pip install -r requirements.txt
```

Run one plain fixed-budget sweep:

```bash
python -m src.sweep \
  --dataset cifar10 \
  --model-family plain \
  --width-schedule stagewise \
  --target-params 250000 \
  --depths 2,4,6,8,10,12 \
  --epochs 30 \
  --seeds 0,1 \
  --measure-flops \
  --track-grad-norm \
  --no-progress
```

Generate figures and tables:

```bash
python -m src.plot_results --summary results/sweep_summary.csv
python -m src.analysis --summary results/sweep_summary.csv
```

## Repository Notes

The assignment PDF, official templates, datasets, local result folders, and local project notes are intentionally ignored and not uploaded to GitHub:

```text
Assignment2-1.pdf
Template/
data/
results/
PROJECT_NOTES.md
```
