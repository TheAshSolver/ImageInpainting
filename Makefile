# ==============================================================================
# QUALCOMM SNAPDRAGON 8 ELITE INPAINTING RESEARCH PIPELINE
# ==============================================================================
# Deterministic, end-to-end reproducible research workflow strictly enforcing
# the research integrity guardrails defined in GEMINI.md.
# ==============================================================================

.PHONY: all audit pilot independence train_routers parity alive_benchmark bakeoff figures audit_deck manifest clean help

all: audit independence train_routers parity alive_benchmark bakeoff figures audit_deck manifest
	@echo "======================================================================"
	@echo "🎉 FULL PUBLICATION RESEARCH PIPELINE REPRODUCED SUCCESSFULLY!"
	@echo "   - Summary metrics, figures, slide decks, and manifest are 100% verified."
	@echo "======================================================================"

help:
	@echo "Available targets:"
	@echo "  make all             : Deterministically reproduce the entire pipeline"
	@echo "  make audit           : Audit immutable raw splits and empirical data"
	@echo "  make pilot           : Run Step 1 Pilot (15 images x 7 arms x 3 repeats) variance benchmark"
	@echo "  make independence    : Run two-way ANOVA image vs system interaction test (R5)"
	@echo "  make train_routers   : Train & benchmark multi-paradigm routers with 95% CIs (R8, R9, R10)"
	@echo "  make parity          : Run 200-sample Android vs Python decision parity test (R11)"
	@echo "  make alive_benchmark : Benchmark on-device 'Alive' loop latency overhead (R12)"
	@echo "  make bakeoff         : Evaluate prompt inpainting bake-off and context ablation (R2, R3)"
	@echo "  make figures         : Generate publication-grade vector PDF/SVG/PNG figures"
	@echo "  make audit_deck      : Adversarial 24-point audit of slide decks against CSVs (R13)"
	@echo "  make manifest        : Generate SHA-256 reproducibility manifest (R14)"

audit:
	@echo ">>> [Phase 1/9] Auditing empirical dataset splits and integrity..."
	python3 -c "import pandas as pd, json; \
		t48 = json.load(open('data/splits/train_4800.json')); \
		assert len(t48['sample_ids']) == 4800, 'Train split count mismatch'; \
		t12 = json.load(open('data/splits/test_1200.json')); \
		assert len(t12['sample_ids']) == 1200, 'Test split count mismatch'; \
		df12 = pd.read_csv('artifacts/test_metrics_1200.csv'); \
		assert len(df12) == 8400, f'Test rows mismatch: {len(df12)} != 8400'; \
		assert not df12.isna().any().any(), 'Found NaNs in test dataset'; \
		print('  ✓ Data split and empirical tables integrity verified.')"

pilot:
	@echo ">>> [Phase 1.5/9] Executing Step 1 Pilot Variance Benchmark..."
	python3 scripts/benchmarks/run_pilot_15_variance.py

independence:
	@echo ">>> [Phase 2/9] Running statistical independence interaction test (R5)..."
	python3 scripts/statistical_interaction_test.py

train_routers:
	@echo ">>> [Phase 3/9] Training & evaluating multi-paradigm routers (R8, R9, R10)..."
	python3 scripts/train_multi_paradigm_routers.py

parity:
	@echo ">>> [Phase 4/9] Executing 200-sample decision parity verification (R11)..."
	python3 scripts/verify_router_decision_parity.py

alive_benchmark:
	@echo ">>> [Phase 5/9] Benchmarking 'Alive' adaptation loop overhead (R12)..."
	python3 scripts/benchmark_alive_loop_overhead.py

bakeoff:
	@echo ">>> [Phase 6/9] Evaluating prompt inpainting bake-off and ablation (R2, R3)..."
	python3 scripts/evaluate_prompt_bakeoff_and_ablation.py

figures:
	@echo ">>> [Phase 7/9] Plotting publication figures (Figures 1-6)..."
	python3 scripts/plot_paper_figures.py

audit_deck:
	@echo ">>> [Phase 8/9] Running adversarial slide deck and claims ledger audit (R13)..."
	python3 scripts/audit_presentation.py

manifest:
	@echo ">>> [Phase 9/9] Generating SHA-256 reproducibility manifest (R14)..."
	python3 scripts/generate_reproducibility_manifest.py

clean:
	rm -f data/processed/*.tmp
	rm -f final_presentation/audit/*.tmp
