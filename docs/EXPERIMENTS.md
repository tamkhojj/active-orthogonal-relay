# Experiments

A mechanistic study of refusal across seven instruction-tuned models. Peak layer
and Causal Basin are discovered automatically per model (`aor/interp/basin.py`);
no per-model hand-tuning is required. One pipeline run regenerates every result;
see [`FINALIZE.md`](FINALIZE.md) for the orchestrator.

## Scripts

Experiments are referenced by code (Exp1-Exp10) in the paper and tables. Each writes
one JSON per model into `results/<model>/`.

| Script | Produces | Summary |
|--------|----------|---------|
| `exp1_sufficiency.py` | exp1 | Sufficiency: adding the generator subspace to benign activations induces refusal; a random axis of equal norm does not. |
| `exp2_recovery_curve.py` | exp2 | Self-repair curve: an early rank-1 ablation is partially rebuilt downstream, and the rebuild is attributed to MLP vs attention writes. |
| `exp3_recovery_operator.py` | exp3 | The downstream rebuild is a low-rank linear operator R; its effective rank is k*, and what it regenerates lies in the generator subspace. |
| `exp4_identification.py` | exp4 | Continuous rank-k ablation shows compensation then collapse; the effect is specific to the generator basis (random/matched-energy controls). |
| `exp5_rrg.py` | exp5 | Discrete Frenet-Serret geometry (curvature, torsion) in the basin vs an out-of-basin window, plus the torsional self-repair coupling onto (e1, e2, e3). |
| `exp6_geora.py` | exp6 | Behavioral, utility, and torsion-bounded ablation evaluation (StrongREJECT mean/binary, non-refusal, utility, forced response). |
| `run_utility_eval.py` | exp7 | Utility and over-refusal benchmarks (TruthfulQA MC1/MC2, XSTest) under the Exp5/Exp6 configuration. |
| `exp8_reviewer_defense.py` | exp8 | Mechanistic controls (group 1): split-half CV, isotropic and covariance-matched nulls, factual and behavioral controls, direct axis knock-out, steering. |
| `exp9_protocol_ablations.py` | exp9 | Protocol and fairness ablations (group 2): full_resid vs writes, optimal position, 512-token parity, system-prompt robustness. |
| `exp10_e3_gold_and_audit.py` | exp10 | Gold-judge direct knock-out of e3 vs e2/e1/random orthogonal, with TruthfulQA and a refusal-marker / Unicode audit. |

### Reviewer-response additions

| Script | Produces | Summary |
|--------|----------|---------|
| `strengthen_stats.py` | stats | 10,000-permutation test with Holm-Bonferroni and per-model bootstrap confidence intervals. |
| `generalize_ood.py` | ood | e3 knock-out on unseen benchmarks (StrongREJECT, JailbreakBench) with no overlap with the estimation data. |
| `repeated_splithalf.py` | splithalf | Stability of the e3 axis across many random split-half partitions. |
| `fill_gaps.py` | patches | Targeted evaluation that fills the two remaining table cells (additive JSON patches). |

## Running

Offline wiring check (no GPU, no downloads):

```bash
python -I aor/experiments/run_pipeline.py --selftest
pytest -q
```

Reproduce one model end-to-end (peak and basin discovered automatically):

```bash
PYTHONNOUSERSITE=1 python -s aor/experiments/run_recovery.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --harmful aor/data/splits/harmful_train.json \
  --benign  aor/data/splits/harmless_train.json \
  --device cuda --outdir results/Llama-3.1-8B-Instruct
```

Reproduce everything (all models, all stages, export, verify):

```bash
bash scripts/run_pipeline.sh
```

To override the automatic selection, pass `--peak N --basin lo-hi`.
