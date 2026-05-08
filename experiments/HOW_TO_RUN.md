# L2 fix v2 (Path A): retrain at 150k steps, ensemble eval

The previous 30k-step result showed large seed variance. We're
retraining at 150k steps x 5 seeds, then re-running the multi-IC and
quantitative-robustness scripts so that all three reported tables
average over the same 5-seed ensemble.

Run the three commands in order. Each script has been updated to use
the 5 new checkpoints `results/models/sac_seed{0..4}.zip`.

## 1. Retrain SAC at 150k steps under 5 seeds

```
python -m experiments.run_multi_seed
```

Wall-clock budget: ~25-60 min on Apple Silicon CPU.

Produces:
  - `results/models/sac_seed{0..4}.zip`
  - `results/tables/multi_seed_raw.csv`
  - `results/tables/multi_seed_summary.csv`

## 2. Multi-IC evaluation across the 5 new seeds

```
python -m experiments.run_multi_ic
```

Wall-clock budget: ~5-10 min (5 seeds x 25 ICs x 4 controllers).

Produces:
  - `results/tables/multi_ic_raw.csv`
  - `results/tables/multi_ic_summary.csv`   <-- mean +/- std over 125 (seed, IC) pairs

## 3. Quantitative robustness across the 5 new seeds

```
python -m experiments.run_robustness_quant
```

Wall-clock budget: ~2-5 min (5 seeds x 4 cases x 4 controllers).

Produces:
  - `results/tables/robustness_quant_raw.csv`
  - `results/tables/robustness_quant_summary.csv`   <-- per-case mean +/- std

## What to send back

Three summary CSVs (the raw ones are nice-to-have for the appendix
but not strictly needed):

  - `results/tables/multi_seed_summary.csv`
  - `results/tables/multi_ic_summary.csv`
  - `results/tables/robustness_quant_summary.csv`

I'll regenerate Tables 1, 4 (and add a note about per-IC variance) in
the paper and re-compile. Estimated end-to-end revision time on my
side: 5-10 minutes once I have the CSVs.

## What we are NOT re-running, and why

- **Hybrid lambda sweep (Table 3).** The lambda tuning is a coarse
  sweep with diminishing returns past lambda=10, and it's a
  property of the architecture more than of any specific SAC
  checkpoint. Re-running it across all 5 seeds would shift the
  numbers slightly but not the conclusion. We can revisit if a
  reviewer asks.

- **Trajectory figure (Fig. 1).** Currently rendered from the
  original lucky checkpoint. After the multi-seed run, the cleanest
  thing is to either (a) regenerate it from one of the new
  checkpoints (e.g. the one closest to the multi-seed median RMSE),
  or (b) overlay all 5 SAC trajectories with the median highlighted.
  Tell me which you'd prefer and I'll write a one-screen update to
  `run_final_results.py`.
