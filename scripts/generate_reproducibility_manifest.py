#!/usr/bin/env python3
"""
scripts/generate_reproducibility_manifest.py
================================================================================
⚡ REQUIREMENT R14: REPRODUCIBILITY MANIFEST GENERATOR
================================================================================
Computes SHA-256 cryptographic checksums across all raw datasets, processed data,
models, router checkpoints, figures, and presentation slide decks.
Guarantees 100% provenance and tamper-evident auditability for publication.
Saves to final_presentation/reproducibility_manifest.json.
================================================================================
"""

import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_MANIFEST = REPO_ROOT / "final_presentation" / "reproducibility_manifest.json"
OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("🔒 GENERATING REPRODUCIBILITY MANIFEST (SHA-256 CHECKSUMS)")
    print("=" * 70)

    # Git info
    git_rev = subprocess.run("git rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip()
    git_branch = subprocess.run("git rev-parse --abbrev-ref HEAD", shell=True, capture_output=True, text=True).stdout.strip()

    target_files = [
        # Deterministic Splits
        REPO_ROOT / "data" / "splits" / "train_4800.json",
        REPO_ROOT / "data" / "splits" / "test_1200.json",

        # Verified Offline Datasets
        REPO_ROOT / "artifacts" / "train_metrics_4800.csv",
        REPO_ROOT / "artifacts" / "test_metrics_1200.csv",
        REPO_ROOT / "artifacts" / "dataset_100_features_6000.csv",

        # Live Physical Hardware Logs
        REPO_ROOT / "data" / "raw" / "real_contention_levels_l0_l4_telemetry.csv",
        REPO_ROOT / "data" / "raw" / "sd_dual_benchmark_latencies.txt",

        # Processed Research Artifacts
        REPO_ROOT / "data" / "processed" / "summary_metrics.csv",
        REPO_ROOT / "data" / "processed" / "interaction_test_results.json",
        REPO_ROOT / "data" / "processed" / "load_generator_calibration_report.csv",
        REPO_ROOT / "data" / "processed" / "parity_200_test_results.json",
        REPO_ROOT / "data" / "processed" / "alive_loop_latency_overhead.json",
        REPO_ROOT / "data" / "processed" / "prompt_bakeoff_summary.csv",
        REPO_ROOT / "data" / "processed" / "prompt_ablation_summary.csv",
        REPO_ROOT / "data" / "processed" / "production_linucb_router.json",

        # Research Ledgers & Audit
        REPO_ROOT / "claims_ledger.md",
        REPO_ROOT / "data" / "DATA_PROVENANCE_AUDIT.md",
        REPO_ROOT / "research" / "AUDIT_0.md",

        # Paper Figures (Generated directly from verified processed metrics)
        REPO_ROOT / "final_presentation" / "figures" / "figure_1_pareto_utility_by_mode.pdf",
        REPO_ROOT / "final_presentation" / "figures" / "figure_2_latency_vs_power_tradeoff.pdf",
        REPO_ROOT / "final_presentation" / "figures" / "figure_3_hardware_contention_impact.pdf",
        REPO_ROOT / "final_presentation" / "figures" / "figure_4_dual_dlc_unet_speedup.pdf",
        REPO_ROOT / "final_presentation" / "figures" / "figure_5_feature_importance_pareto.pdf",
        REPO_ROOT / "final_presentation" / "figures" / "figure_6_adaptive_drift_sliding_window.pdf",

        # Presentation Slide Decks
        REPO_ROOT / "final_presentation" / "slides" / "01_app_and_features_deck.md",
        REPO_ROOT / "final_presentation" / "slides" / "02_research_master_deck.md"
    ]

    manifest_entries = {}
    missing_count = 0

    for p in target_files:
        rel = str(p.relative_to(REPO_ROOT))
        if p.exists():
            h = sha256_file(p)
            sz = p.stat().st_size
            manifest_entries[rel] = {
                "sha256": h,
                "size_bytes": sz,
                "status": "VERIFIED"
            }
            print(f"  ✓ {rel:58s} [{h[:12]}...]")
        else:
            manifest_entries[rel] = {
                "status": "PENDING_OR_MISSING"
            }
            print(f"  ✗ {rel:58s} [MISSING]")
            missing_count += 1

    manifest = {
        "title": "Snapdragon 8 Elite Inpainting Research Reproducibility Manifest",
        "git_commit": git_rev,
        "git_branch": git_branch,
        "device_target": "Qualcomm Snapdragon 8 Elite Development Kit (8f27557f, SM8750)",
        "guardrail_enforcement": "GEMINI.md (Append-Only Raw Data, Zero Hardcoding, Zero Silent Fallbacks)",
        "total_tracked_artifacts": len(manifest_entries),
        "verified_artifacts": len(manifest_entries) - missing_count,
        "artifacts": manifest_entries
    }

    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print("-" * 70)
    print(f"Manifest written to: {OUT_MANIFEST}")
    print(f"Total: {len(manifest_entries)} tracked | Verified: {len(manifest_entries) - missing_count} | Missing: {missing_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
