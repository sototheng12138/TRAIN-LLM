# Dataset layout

## Primary file (Iron freight, paper experiments)

- **`2023_2025_Iron_data.csv`**
  - Column `date` or `日期` (time index)
  - Four multivariate targets: `铁矿石`, `铁矿砂`, `铁矿粉`, `铁精矿粉`
  - Daily frequency; sorted ascending by date in loader

## Optional M5-style matrices

- `m5.csv`, `m5_10ch.csv`, etc. — used in some few-shot / M5 experiments; format documented in `run_baseline_xgb_arima.py` (`--data_format m5_matrix`).

## Prompt bank

- `prompt_bank/` — text assets for domain prompts when enabled in training.

## Redistribution

Public hosting requires **legal clearance** for the CSV. Alternatives: a **synthetic sample** or **checksum-only** placeholder plus a **data availability** statement in the paper.
