## Repo Audit

### What already exists

- `../FedGate/cbms_repro` contains the closest existing centralized baseline to the target paper.
- `../FedGate/cbms_repro/model.py` already implements:
  - a 3D ViT MRI encoder
  - an FT-Transformer tabular encoder
  - a multimodal fusion/classification stack
- `../FedGate/cbms_repro/data.py` already implements:
  - subject-level row loading from `cbms_dataset.csv`
  - train-only median imputation
  - train-only standardization
  - MRI loading and resize
- `../FedGate/cbms_repro/train.py` already implements:
  - end-to-end centralized training
  - 5-fold CV
  - multi-seed runs
  - AdamW, cosine annealing, early stopping, gradient clipping

### What exists but is not the target

- `../FedGate/cbms_fed` contains a federated pipeline on raw data, but its model is not aligned with the paper baseline:
  - MRI encoder is a small CNN, not a 3D ViT
  - tabular encoder is an MLP, not an FT-Transformer
  - therefore it is not yet a fair FedAvg/FedGate counterpart to the paper-like centralized model
- Historical `results/` and export scripts in `FedGate_final` assume older experiment outputs. They are useful for aggregation style, but not yet wired to a paper-like training stack.

### FedAvg / FedGate status today

- FedAvg exists only indirectly through the Flower infrastructure in `../FedGate/cbms_fed`.
- FedGate exists only in the simplified local architecture from `../FedGate/cbms_fed`.
- No current federated implementation in `FedGate_final` matches the required rule:
  - standard FL baseline: same multimodal stack as centralized, globally aggregated with FedAvg
  - FedGate: local encoders + local gating, only the final classifier is shared with FedAvg aggregation

### Missing pieces versus `todo.txt`

- No YAML-driven centralized launcher in `FedGate_final`
- No standardized `metrics.json`, `summary.csv`, `config_used.yaml` contract for centralized runs
- No client-level metrics pipeline yet
- No fair paper-aligned FedAvg baseline yet
- No paper-aligned FedGate pipeline with classifier-only sharing yet
- Scenario configs `S0`-`S3` exist, but are not yet connected to the paper-like centralized/FL stack

### Practical conclusion

- The fastest correct first step is to reuse `../FedGate/cbms_repro` as the centralized scientific baseline and wrap it cleanly from `FedGate_final`.
- The federated part should then be rebuilt in `FedGate_final` around the same multimodal encoders/fusion logic, instead of reusing the simplified CNN/MLP stack as-is.
- The target methodology is now:
  - centralized pooled baseline with the same multimodal backbone as standard FL
  - standard FL baseline trained with FedAvg
  - FedGate trained with FedAvg too, but sharing only the final classifier
