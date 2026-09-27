# Data

Place the cleaned SGCC-derived CSV here as `cleaned_data.csv`.

The dashboard can handle the source-file layout used in this project (date columns first, `CONS_NO` and `FLAG` at the end) as well as the more conventional `CONS_NO`, `FLAG`, then date columns.

For Review 1, the consumer-analysis page treats the final 30 calendar days as the target period and the preceding windows as historical features. This mirrors the notebook setup.
