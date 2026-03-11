## Implementation Plan

### Step 1: Centralized baseline in `FedGate_final`

Files to create:
- `scripts/run_centralized_baseline.py`
- `scripts/run_centralized_baseline.sh`
- `configs/centralized_multimodal.yaml`
- `AUDIT.md`
- `IMPLEMENTATION_PLAN.md`
- `src/models/paper_multimodal.py`
- `src/training/centralized.py`

Files to modify:
- central/federated scripts under `FedGate_final`

Decisions:
- Centralized training now lives directly in `FedGate_final`.
- The centralized architecture is aligned with the standard FL baseline:
  - ViT encoder
  - FT-Transformer encoder
  - simple concatenation fusion
  - MLP classifier
- Add richer metrics and checkpoints so `FedGate_final` can validate and standardize outputs.

Centralized output contract:
- `metrics.json`
- `summary.csv`
- `config_used.yaml`
- `raw_results.json`
- `checkpoints/seed_<seed>_fold_<fold>_best.pt`
- `checkpoints/seed_<seed>_fold_<fold>_final.pt`

### Step 2: Fair FedAvg baseline

Planned files:
- `src/models/paper_multimodal.py`
- `src/data/cbms_data.py`
- `src/federated/fedavg.py`
- `scripts/run_fedavg.py`

Decision:
- FedAvg will use the same multimodal encoders and simple concatenation fusion as the centralized baseline.
- Shared parameter scope will be explicit in code and config.
- No gating-specific personalization will be enabled in FedAvg.

Main risk:
- Full end-to-end federation of the paper-sized ViT may be heavy on a remote server. If memory becomes the blocker, the exact scope that remains federated will have to be documented explicitly.

### Step 3: FedGate variant

Planned files:
- `src/models/fedgate_multimodal.py`
- `src/federated/fedgate.py`
- `scripts/run_fedgate.py`

Decision:
- ViT encoder and FT-Transformer encoder stay local on clients.
- Gating stays local on clients.
- Only the classifier head is exchanged and aggregated with FedAvg.

Main risk:
- The current repo does not yet contain a paper-aligned federated multimodal model with classifier-only sharing, so this part is a real implementation effort, not a thin wrapper.

### Step 4: Scenarios and exports

Planned files:
- scenario-aware loaders under `src/data`
- run scripts for `S0`-`S3`
- export scripts updated to ingest the new output contract

Decision:
- Existing scenario YAMLs in `configs/` remain the source of truth for split definitions.
- Export scripts will be adapted after the new metrics format stabilizes.
