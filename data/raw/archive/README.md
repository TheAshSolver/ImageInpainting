# RAW TELEMETRY ARCHIVE & PROVENANCE LEDGER

This directory archives all historical empirical raw telemetry datasets generated during Qualcomm Snapdragon 8 Elite HDK benchmark sweeps. In accordance with Rule 1 of GEMINI.md, raw logs are strictly append-only and read-only.

---

## Historical Files & Incident Log

### 1. Initial 150-Trial Sweep (`eef5ce6d...`) [LOST / OVERWRITTEN INCIDENT]
- **Original Path**: `data/raw/real_orchestrated_contention_telemetry.csv`
- **Recorded SHA-256**: `eef5ce6d35c6e6be93a918f2a777d9b70fce20456c47cf8eb6361af4362f3687`
- **Trial Count**: 150 trials ($N=10$ per config $\times 15$ states), 3,750 samples.
- **Incident Report**: On 2026-11-19, executing `orchestrator.py --fresh` unintentionally executed an ADB `rm -f /data/local/tmp/real_contention_telemetry.csv` and local `os.remove(RAW_OUT_CSV)` before archiving the file. This violated Rule 1 (Immutable raw logs).
- **Corrective Measure**: All orchestrator scripts are now permanently barred from deleting or overwriting any raw file. Future sweeps must write to unique timestamped files: `telemetry_<timestamp>_<gitcommit>.csv`.

### 2. Mixed Codebase 2-Trial Calibration Run (`46826b...`)
- **Archived Path**: `data/raw/archive/real_orchestrated_contention_telemetry_2trial_mixed_46826b.csv`
- **SHA-256**: `46826bcdc2e05551d7ec18647a828e5b9907013dbd4ab488b476e37b6ad1c091`
- **Trial Count**: 30 trials ($N=2$ per config $\times 15$ states), 742 samples.
- **Git Commit Range**: `3742d4c` (with intermediate controller PID tuning).
- **Status**: Quarantined as mixed code-version calibration telemetry.

---

## Strict Policy Enforcement (Rule 1 & Rule 4)
- No script may delete or overwrite files in `data/raw/`.
- Every row in all future sweeps must include `git_commit`, `binary_hash`, and live per-core frequencies (`cpu0_freq_mhz`..`cpu7_freq_mhz`).
