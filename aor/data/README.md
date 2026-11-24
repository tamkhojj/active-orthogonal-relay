# Data

Only the data the experiments actually read is shipped here. All experiments load
the released splits in [`splits/`](splits/) directly, so the results reproduce
without rebuilding anything.

## Layout

| Path | Contents |
|------|----------|
| `splits/` | The released train / test splits used in every experiment. `harmful_*` and `harmless_*` are minimal-paired (one benign counterpart per harmful instruction). Train = 260, test = 572, 0% overlap. |
| `processed/` | The two out-of-distribution evaluation benchmarks: `strongreject.json` and `jailbreakbench.json`. |

## Provenance

The harmful instructions in the splits are drawn from three public safety
benchmarks; each is paired with a matched benign counterpart to form the harmless
set. These source corpora are not shipped (the splits already contain the final
prompts); they are listed here for provenance.

| Dataset | Role | Source |
|---------|------|--------|
| AdvBench | harmful source for the splits | Zou et al., 2023 |
| MaliciousInstruct | harmful source for the splits | Huang et al., 2024 |
| HarmBench | harmful source for the splits | Mazeika et al., 2024 |
| StrongREJECT | judge + OOD evaluation (shipped) | Souly et al., 2024 |
| JailbreakBench | OOD generalization (shipped) | Chao et al., 2024 |

Behavioral refusal is scored with the open `strongreject_finetuned` judge; XSTest
and TruthfulQA (loaded on demand from the Hugging Face Hub) measure over-refusal
and utility. All datasets are used under their original licenses for
reproducibility only.
