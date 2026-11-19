# RESEARCH INTEGRITY & ENGINEERING GUARDRAILS

Every autonomous agent, subagent, and human engineer operating in this repository must strictly adhere to these four non-negotiable research integrity rules:

---

## Rule 1: Immutable Raw Experiment Logs
- **`data/raw/` and `data/logs/` are append-only and strictly read-only.**
- Under no circumstances may an agent, parser, or script modify, overwrite, truncate, or delete raw experiment output files, hardware telemetry logs, or stdout traces.
- All derived transformations must write to `data/processed/` without altering source logs.

## Rule 2: No Silent Fallbacks (Crash Loudly)
- **Banned**: `try/except` blocks that catch exceptions and silently substitute `0`, `None`, fallback strings, or interpolated averages.
- **Banned**: Silently dropping malformed rows, missing timestamps, or incomplete trials without terminating execution.
- Every ingestion script, telemetry parser, and evaluator must **fail fast and loudly** (`assert` / `raise ValueError`) if:
  1. An expected experiment run, file, or trial column is missing.
  2. Any metric evaluates to `NaN`, `Inf`, or uncalibrated zero (e.g. `latency <= 0`, `power <= 0`).
  3. Trial counts per configuration do not match the expected sample size ($N$).

## Rule 3: Zero Hardcoded Numbers in Visuals and Slides
- **Strictly Forbidden**: Hardcoding statistical metrics, percentages, p-values, latency numbers, or energy readings inside plotting scripts, presentation generators, HTML templates, or Markdown slides.
- **Single Source of Truth**: Every number, axis limit, error bar, and table cell must be dynamically loaded from verified processed data (e.g., `data/processed/summary_metrics.csv` or `claims_ledger.md`).
- Every claim presented on a slide must trace back to a specific row, column, and mathematical formula recorded in `claims_ledger.md`.

## Rule 4: Fail-Fast Assertions & Deterministic Data Splits
- All data preparation and evaluation scripts must include explicit assertion barriers:
  ```python
  assert len(df) == EXPECTED_TRIALS, f"Trial count mismatch: expected {EXPECTED_TRIALS}, got {len(df)}"
  assert not df.isna().any().any(), "Found NaN values in dataset"
  assert (df['latency_ms'] > 0).all(), "Found non-positive latency values"
  ```
- **Data Splits**: `data/splits/train_4800.json` and `data/splits/test_1200.json` are generated once with a fixed seed (`seed=42`) and locked with SHA-256 checksums before training or evaluating any routing model.
- Evaluation on `test_1200` must have zero data leakage: scalers, priors, and feature encoders must be fit exclusively on `train_4800`.

---

## Hardware & QIDK Disconnect Resilience
- The Qualcomm Snapdragon 8 Elite Development Kit (`8f27557f`) is prone to USB PHY resets under sustained high-bandwidth transfers.
- **Rate Limiting**: All ADB transfers must be throttled ($\le 12\text{ MB/s}$) to prevent USB buffer overruns.
- **State Checkpoints**: Every benchmarking harness must write an atomic `checkpoint.json` after every batch.
- **Recovery Ledger**: Always check and update `QIDK_RECOVERY_LEDGER.md` so work resumes exactly from the last verified checkpoint if a disconnect occurs.
