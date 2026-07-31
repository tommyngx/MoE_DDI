# Master instructions for coding agents

## Mission

Implement and validate MoEDDI: a feature-family-aware Mixture-of-Experts model
for 178-class drug-drug interaction prediction, compared fairly with a
T-DDI-style numerical MLP on the same 3,780 descriptors and splits.

## Non-negotiable data safety

- `Dataset/` contains local, large, unversioned data. Never edit, rename,
  recompress, delete, or commit anything under it.
- Keep `/Dataset/` in the root `.gitignore`. Before every commit, run
  `git check-ignore Dataset/train_extracted.csv` and inspect `git status`.
- Do not create derived files inside `Dataset/`.
- Write every experiment to `runs/<run_name>/`: `best.pt` and `last.pt` at the
  run root, metadata under `info/`, and plots under `plots/`; all are ignored.
- Never load a complete split into RAM. The three CSVs total about 19 GB.

## Dataset contract

- Discover columns by header name.
- Expected target: `class`.
- Expected metadata:
  `drugid-drug_a`, `drugid-drug_b`, `drugname-drug_a`,
  `drugname-drug_b`, `drugsmiles-drug_a`, `drugsmiles-drug_b`.
- All other columns are numeric descriptor features.
- The current schema has exactly 3,780 features and 178 classes. Raw class IDs
  are non-contiguous (currently 1–218), so use the persisted label vocabulary.
- Accept integral labels serialized as either integers or floats (for example,
  `1` and `1.0`), map them through the train-derived vocabulary, and reject
  unknown or non-integral labels.
- Fit normalization and imbalance weights on training data only.

## Scientific fairness

- Compare models using identical input columns, split membership, preprocessing,
  seeds, epoch budget, selection metric, and evaluation code.
- Report accuracy, macro/micro/weighted F1, per-class F1/support, top-k
  accuracy, and calibration on the complete held-out test set.
- Treat `configs/tddi.yaml` as a configurable reproduction profile, not proof
  of exact paper reproduction until parameter counts and all hyperparameters
  have been checked against the authors' released code.
- Never compare a full-test score with a filtered high-confidence score.
- Run at least three seeds for final tables and report mean ± standard deviation.

## Architecture rules

- Feature-family membership is derived deterministically from descriptor names.
- Cross-family descriptors may belong to both relevant experts; document this
  overlap in every run's schema artifact.
- The router must expose probabilities for specialization analysis.
- Include load-balancing loss in MoE training and log it separately.
- Keep Python modules flat directly under `src/`; do not create a package or
  model subdirectory. The principal model class is `MoEDDI` in `src/models.py`.
- Every new architecture needs shape, forward/backward, serialization, and
  deterministic inference tests.

## Split rules

- Existing train/validation/test files are the official random split.
- Strict unseen-drug split: assign each drug to exactly one partition and retain
  only pairs whose two drugs belong to the same partition. Mark cross-partition
  pairs as dropped.
- Scaffold split: derive Bemis-Murcko scaffold per unique drug, assign scaffold
  groups to partitions, and retain only within-partition pairs.
- Store split assignments as compact manifests; never duplicate the feature CSVs.
- Audit drug/scaffold leakage, class coverage, retained sample counts, and
  dropped-pair rate before training.

## Required checks before hand-off

```bash
python -m pytest
python -m ruff check .
moeddi inspect --config configs/smoke.yaml
python src/train.py --config configs/smoke.yaml --data Dataset --force-stats
python src/evaluate.py --config configs/smoke.yaml --data Dataset \
  --run-dir runs/smoke --checkpoint runs/smoke/best.pt
```

For changes that affect full experiments, also run a bounded real-data trial and
record the exact config and outcome. Do not claim full-dataset performance from
a smoke run.

## Definition of done

- Tests pass.
- The bounded run finishes without loading full CSVs into memory.
- Artifacts contain resolved config, schema, statistics, checkpoints, history,
  aggregate metrics, and per-class metrics.
- Dataset files remain ignored and unchanged.
- Limitations and any deviations from the T-DDI reference are explicit.
