# T-DDI implementation audit

This audit compares the MoE_DDI pipeline with the local released code in
`Reference/tddi-main`. It distinguishes the active numerical model from unused
TabTransformer parameters and the single-split pipeline from the paper's ensemble protocol.

## Confirmed released configuration

- Input: 3,780 numerical descriptors after dropping six identity columns by name.
- Active model path: `LayerNorm(3780) -> 7560 -> ReLU -> 7560 -> ReLU -> 178`.
- Loss: unweighted focal loss with `gamma=1.0`.
- Optimizer: AdamW, learning rate `9.4526e-5`, weight decay `1.5446e-4`.
- Schedule: cosine annealing with `T_max=50`.
- Training: up to 200 epochs, patience 50, best fold selected by validation accuracy.
- Protocol: concatenate the original train and validation sets, create three stratified folds,
  train one model per fold, and average their class probabilities on the untouched test set.
- Paper full-test result: accuracy 0.9434 and macro-F1 0.8452. The reported 0.9796 accuracy
  is only for the 87.91% high-confidence subset and must not be compared with a full test score.

The released module reports 87,448,646 parameters because it still instantiates categorical
embedding/transformer parameters even though there are no categorical inputs. Those parameters
are never called. `TDDINumericalMLP` implements only the active 87,098,938-parameter numerical
path, so omitting the dead branch does not alter logits or gradients of the active head.

## Does the released repository contain another predictive model?

No. `arch/models.py` exposes one DDI predictor, `TDDI_Model`, which wraps the
`TabTransformer` implementation. With zero categorical columns its attention branch is inactive,
leaving the numerical MLP above as the only predictive architecture used for the paper result.
The repository's uncertainty estimator averages the three fold members; LIME explains their
predictions; and PyBioMed computes descriptors. None of those components is a second classifier.

## Gaps found in the previous repository version

| Area | Previous repository | Released T-DDI | Likely impact |
|---|---|---|---|
| MLP shape | 3780 -> 11340 -> 3780 -> 178 | 3780 -> 7560 -> 7560 -> 178 | High |
| Input transform | Per-feature train mean/std, then LayerNorm | Model LayerNorm only | High |
| Focal gamma | 2.0 | 1.0 | Medium/high |
| Epoch budget | 50 | 200 | High |
| Weight decay | 0 | 1.5446e-4 | Medium |
| Scheduler | cosine over total epochs | cosine `T_max=50` | Medium |
| Seed | 2026 | 42 | Reproducibility |
| Checkpoint metric | macro-F1 | accuracy | Reproducibility |
| Evaluation | one train/validation model | 3-fold probability ensemble | High |
| Confidence | max-probability ECE only | normalized entropy plus ensemble variance/MI | Reporting |

Together, these differences are sufficient to explain a two-to-three percentage point gap;
the gap should not be attributed to missing descriptors before rerunning the corrected baseline.

## Changes made

- `configs/tddi.yaml` now matches all active single-model hyperparameters above.
- Normalization is configurable; T-DDI and the new hybrid both use `none` so the embedded global
  backbone receives the same raw descriptors before its internal LayerNorm.
- Evaluation can average any number of checkpoints in one streaming pass and reports predictive
  entropy, ensemble variance, mutual information, and normalized entropy confidence.
- The main MoE profile is now a residual hybrid: the complete active T-DDI backbone supplies
  global logits and a descriptor representation, while chemical-family experts learn additive
  corrections. It has 97,965,407 parameters versus T-DDI's 87,098,938 active parameters; the old
  MoE had only about 6.7M and compressed all 3,780 features to 256 dimensions too early.
- The residual classifier is zero-initialized. Loading a corrected T-DDI checkpoint therefore
  produces bit-for-bit equivalent initial logits. Epoch-0 validation is saved as an eligible
  `best.pt`, so hybrid checkpoint selection can fall back to the imported T-DDI baseline.
- The imported T-DDI path may be frozen for a warm-up and fine-tuned with a lower learning rate.
  Auxiliary losses keep both the global and family-MoE branches directly predictive.
- Dense routing gives every descriptor family a classification gradient without additional FLOPs,
  because all experts are already evaluated in this implementation.
- The main profile adopts T-DDI's focal gamma and optimizer/scheduler settings.
- `configs/moeddi_light.yaml` keeps the earlier 6.8M shared-trunk model, while
  `configs/moeddi_legacy.yaml` preserves the original pre-audit profile.
- Statistics caches now live under `runs/_cache/`, never inside the raw `Dataset/` directory.

## Recommended T-DDI-to-MoE training path

First train one corrected T-DDI on the official train/validation split. This checkpoint is a clean
initialization for the hybrid; do not use a CV member whose training fold may contain rows from the
official validation set used to select the hybrid.

```bash
python src/train.py \
  --config configs/tddi.yaml \
  --data Dataset \
  --run-name tddi_single_seed42

python src/train.py \
  --config configs/moeddi.yaml \
  --data Dataset \
  --run-name moeddi_hybrid_seed42 \
  --seed 42 \
  --tddi-pretrain runs/tddi_single_seed42/best.pt \
  --freeze-tddi-epochs 5 \
  --tddi-backbone-lr-multiplier 0.2
```

The first command is for weight transfer and a same-split ablation. Continue to use the dedicated
three-fold command below when reproducing the paper's reported T-DDI ensemble.

## Reproducing the three-fold protocol

The ordinary `src/train.py` command intentionally remains a single-model train/validation runner.
The dedicated CV runner constructs three sklearn-compatible stratified fold assignments over the
concatenated train+validation labels, streams the corresponding feature rows without duplicating
CSV files, trains all members, averages probabilities, and writes the ensemble as `best.pt`:

```bash
python src/tddi_cv.py \
  --config configs/tddi.yaml \
  --data Dataset \
  --run-dir runs/tddi_paper_seed42
```

The generic evaluator also supports a deep ensemble of separately trained checkpoints:

```bash
python src/evaluate.py \
  --config configs/moeddi.yaml \
  --checkpoint runs/moeddi_seed42/best.pt \
               runs/moeddi_seed43/best.pt \
               runs/moeddi_seed44/best.pt \
  --run-dir runs/moeddi_ensemble
```

This is useful for MoE uncertainty and often improves accuracy, but three independent seeds on the
same split are not methodologically identical to T-DDI's CV folds. The 0.88 high-confidence
threshold is frozen for the released T-DDI ensemble; MoE needs its own development/OOF threshold
and must not reuse 0.88 as if it were calibrated.

Two implementation-level deviations remain explicit: CSV training shuffles rows inside streaming
record batches rather than constructing one in-memory global permutation, and the CV runner applies
the frozen 0.88 threshold rather than regenerating the paper's OOF threshold-sweep artifacts. These
do not change feature membership or the full-test prediction definition, but they can affect exact
bit-level/epoch-level reproduction.
