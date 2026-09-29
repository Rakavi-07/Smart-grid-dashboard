# Data folder

Place your cleaned SGCC-derived dataset here as:

    data/cleaned_data.csv

Expected format:
- One row per consumer.
- Columns: `CONS_NO`, `FLAG`, then one column per date
  (e.g. `2016-10-02`, `2016-10-03`, ... `2016-10-31`, or however many
  days of history you want available for the rolling-window features).
- `FLAG` is the ground-truth label (0 = normal, 1 = suspicious) — only
  used for display/comparison, never fed into the model at inference time.

If this file is not present, the dashboard automatically falls back to
a small synthetic DEMO dataset (see `utils/data_utils.py`) so it still
runs end-to-end for a live demo or review.
