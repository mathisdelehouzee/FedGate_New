"""Centralized training for the pooled multimodal baseline."""

from __future__ import annotations

import copy
import gc
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from fedgate_final.data.cbms import ScenarioDataset, TabularScaler, build_mri_transforms, load_cbms_csv
from fedgate_final.models.paper_multimodal import PaperFedAvgModel
from fedgate_final.training.metrics import compute_binary_metrics, mean_std_across_runs, write_csv
from fedgate_final.training.progress import make_progress, progress_write
from fedgate_final.training.runtime import apply_runtime_settings


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _pick_device(value: str) -> torch.device:
    text = str(value).strip().lower()
    if text in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_tuple3(value: Any, default: Sequence[int] = (128, 128, 128)) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) == 3:
            return tuple(int(part) for part in parts)
    return tuple(int(v) for v in default)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _compute_class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    total = counts.sum()
    weights = total / (counts + 1e-6)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _loader_kwargs(*, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, prefetch_factor: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return kwargs


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            logits = model(batch["mri"], batch["tabular"], batch["mri_mask"], batch["tab_mask"])
            loss = criterion(logits, batch["label"])
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            y_true.extend(batch["label"].detach().cpu().numpy().tolist())
            y_prob.extend(probs.tolist())
            total_examples += int(batch["label"].size(0))
            total_loss += float(loss.item()) * int(batch["label"].size(0))
    return compute_binary_metrics(np.asarray(y_true), np.asarray(y_prob), total_loss / max(1, total_examples))


def _shutdown_loader(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()
    loader._iterator = None


def _build_model(
    *,
    model_cfg: Dict[str, Any],
    num_features: int,
    mri_shape: tuple[int, int, int],
    hidden_dim: int,
    dropout: float,
    device: torch.device,
) -> PaperFedAvgModel:
    return PaperFedAvgModel(
        num_features=num_features,
        img_size=mri_shape,
        patch_size=tuple(int(v) for v in model_cfg.get("patch_size", (16, 16, 16))),
        mri_dim=int(model_cfg.get("mri_dim", 768)),
        token_dim=int(model_cfg.get("token_dim", 64)),
        projection_dim=hidden_dim,
        attn_dim=hidden_dim,
        vit_layers=int(model_cfg.get("vit_layers", 12)),
        vit_heads=int(model_cfg.get("vit_heads", 12)),
        tab_layers=int(model_cfg.get("tab_layers", 3)),
        tab_heads=int(model_cfg.get("tab_heads", 4)),
        dropout=dropout,
    ).to(device)


def _checkpoint_paths(checkpoints_dir: Path, seed: int, fold_idx: int) -> tuple[Path, Path, Path]:
    stem = f"seed_{seed}_fold_{fold_idx}"
    return (
        checkpoints_dir / f"{stem}_best.pt",
        checkpoints_dir / f"{stem}_final.pt",
        checkpoints_dir / f"{stem}_result.json",
    )


def _load_existing_fold_result(result_path: Path) -> Dict[str, Any] | None:
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def run_centralized_experiment(
    *,
    config_path: Path,
    output_root: Path,
    seeds_override: List[int] | None = None,
    device_override: str = "",
    epochs_override: int = 0,
    batch_size_override: int = 0,
    folds_override: int = 0,
    num_workers_override: int = -1,
    limit_rows_override: int = -1,
    resume_existing: bool = True,
) -> Dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data_cfg = dict(cfg.get("data", {}))
    central_cfg = dict(cfg.get("centralized", {}))
    runtime_cfg = dict(cfg.get("runtime", {}))
    experiment = str(cfg.get("experiment", "centralized_multimodal"))
    output_name = str(central_cfg.get("output_name", experiment))

    csv_path = _resolve_path(config_path.parent, str(data_cfg["csv"]))
    data_root = _resolve_path(config_path.parent, str(data_cfg["data_root"]))
    rows = load_cbms_csv(csv_path, data_root=data_root)
    if limit_rows_override > 0:
        rows = rows[:limit_rows_override]

    seeds = seeds_override if seeds_override is not None else [int(seed) for seed in cfg.get("seeds", [42, 123, 456])]
    device = _pick_device(device_override or str(central_cfg.get("device", "auto")))
    epochs = int(epochs_override or central_cfg.get("epochs", 100))
    batch_size = int(batch_size_override or central_cfg.get("batch_size", 4))
    folds = int(folds_override or central_cfg.get("folds", 5))
    num_workers = int(num_workers_override if num_workers_override >= 0 else central_cfg.get("num_workers", 0))
    lr = float(central_cfg.get("lr", 2e-5))
    weight_decay = float(central_cfg.get("weight_decay", 0.01))
    patience = int(central_cfg.get("patience", 20))
    min_epochs = int(central_cfg.get("min_epochs", 40))
    warmup_epochs = int(central_cfg.get("warmup_epochs", 5))
    hidden_dim = int(cfg.get("model", {}).get("hidden_dim", 128))
    dropout = float(cfg.get("model", {}).get("dropout", 0.1))
    model_cfg = dict(cfg.get("model", {}))
    mri_shape = _to_tuple3(data_cfg.get("mri_shape", (128, 128, 128)))
    mri_cache_cfg = dict(data_cfg.get("mri_cache", {}))
    mri_cache_enabled = bool(mri_cache_cfg.get("enabled", True))
    mri_cache_max_items = int(mri_cache_cfg.get("max_items", 1024))
    mri_cache_dir = (
        _resolve_path(config_path.parent, str(mri_cache_cfg["dir"]))
        if mri_cache_cfg.get("dir")
        else None
    )
    prefetch_factor = int(central_cfg.get("prefetch_factor", 4))
    apply_runtime_settings(runtime_cfg)

    experiment_dir = output_root / output_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    labels = [row.label for row in rows]
    transforms = build_mri_transforms(mri_shape)
    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        set_seed(seed)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            trainval_rows = [rows[int(i)] for i in trainval_idx]
            test_rows = [rows[int(i)] for i in test_idx]
            train_rows, val_rows = train_test_split(
                trainval_rows,
                test_size=0.1,
                stratify=[row.label for row in trainval_rows],
                random_state=seed,
            )

            scaler = TabularScaler()
            scaler.fit(train_rows)
            train_ds = ScenarioDataset(
                rows=train_rows,
                tabular_scaler=scaler,
                mri_transforms=transforms,
                client_id=-1,
                client_type="both_aligned",
                split_name="train",
                scenario_cfg={"apply_to_val": False, "apply_to_test": False},
                seed=seed,
                mri_shape=mri_shape,
                mri_cache_enabled=mri_cache_enabled,
                mri_cache_max_items=mri_cache_max_items,
                mri_cache_dir=mri_cache_dir,
            )
            val_ds = ScenarioDataset(
                rows=val_rows,
                tabular_scaler=scaler,
                mri_transforms=transforms,
                client_id=-1,
                client_type="both_aligned",
                split_name="val",
                scenario_cfg={"apply_to_val": False, "apply_to_test": False},
                seed=seed,
                mri_shape=mri_shape,
                mri_cache_enabled=mri_cache_enabled,
                mri_cache_max_items=mri_cache_max_items,
                mri_cache_dir=mri_cache_dir,
            )
            test_ds = ScenarioDataset(
                rows=test_rows,
                tabular_scaler=scaler,
                mri_transforms=transforms,
                client_id=-1,
                client_type="both_aligned",
                split_name="test",
                scenario_cfg={"apply_to_val": False, "apply_to_test": False},
                seed=seed,
                mri_shape=mri_shape,
                mri_cache_enabled=mri_cache_enabled,
                mri_cache_max_items=mri_cache_max_items,
                mri_cache_dir=mri_cache_dir,
            )
            train_loader = DataLoader(
                train_ds,
                **_loader_kwargs(
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                    prefetch_factor=prefetch_factor,
                ),
            )
            val_loader = DataLoader(
                val_ds,
                **_loader_kwargs(
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                    prefetch_factor=prefetch_factor,
                ),
            )
            test_loader = DataLoader(
                test_ds,
                **_loader_kwargs(
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                    prefetch_factor=prefetch_factor,
                ),
            )
            best_checkpoint_path, final_checkpoint_path, result_checkpoint_path = _checkpoint_paths(checkpoints_dir, seed, fold_idx)

            model: nn.Module | None = None
            optimizer: torch.optim.Optimizer | None = None
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
            criterion: nn.Module | None = None
            try:
                model = _build_model(
                    model_cfg=model_cfg,
                    num_features=len(train_rows[0].features),
                    mri_shape=mri_shape,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    device=device,
                )
                class_weights = _compute_class_weights([row.label for row in train_rows], device=device)
                criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

                if resume_existing and final_checkpoint_path.exists():
                    existing = _load_existing_fold_result(result_checkpoint_path) or {}
                    state_dict = torch.load(final_checkpoint_path, map_location=device)
                    model.load_state_dict(state_dict)
                    test_metrics = _evaluate(model, test_loader, device, criterion)
                    result = {
                        "seed": seed,
                        "fold": fold_idx,
                        "best_epoch": existing.get("best_epoch"),
                        "best_val_auprc": existing.get("best_val_auprc"),
                        "test": test_metrics,
                        "history": existing.get("history", []),
                        "resumed": True,
                    }
                    all_results.append(result)
                    result_checkpoint_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    print(
                        f"[centralized][seed={seed}][fold={fold_idx}] "
                        f"skip existing checkpoint={final_checkpoint_path}"
                    )
                    continue

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

                history: List[Dict[str, Any]] = []
                best_val = float("-inf")
                best_epoch = 0
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0

                epoch_progress = make_progress(
                    range(1, epochs + 1),
                    total=epochs,
                    desc=f"centralized seed={seed} fold={fold_idx}",
                    leave=True,
                )
                try:
                    for epoch in epoch_progress:
                        model.train()
                        total_loss = 0.0
                        total_examples = 0
                        y_true: List[int] = []
                        y_prob: List[float] = []
                        if epoch <= warmup_epochs:
                            warmup_scale = epoch / max(1, warmup_epochs)
                            for group in optimizer.param_groups:
                                group["lr"] = lr * warmup_scale
                        for raw_batch in train_loader:
                            batch = _move_batch(raw_batch, device)
                            optimizer.zero_grad()
                            logits = model(batch["mri"], batch["tabular"], batch["mri_mask"], batch["tab_mask"])
                            loss = criterion(logits, batch["label"])
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                            y_true.extend(batch["label"].detach().cpu().numpy().tolist())
                            y_prob.extend(probs.tolist())
                            total_examples += int(batch["label"].size(0))
                            total_loss += float(loss.item()) * int(batch["label"].size(0))
                        if epoch > warmup_epochs:
                            scheduler.step()

                        train_metrics = compute_binary_metrics(np.asarray(y_true), np.asarray(y_prob), total_loss / max(1, total_examples))
                        val_metrics = _evaluate(model, val_loader, device, criterion)
                        test_metrics = _evaluate(model, test_loader, device, criterion)
                        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "test": test_metrics})
                        epoch_progress.set_postfix(
                            train_acc=f"{train_metrics['acc']:.3f}",
                            val_acc=f"{val_metrics['acc']:.3f}",
                            test_acc=f"{test_metrics['acc']:.3f}",
                            train_loss=f"{train_metrics['loss']:.3f}",
                            val_loss=f"{val_metrics['loss']:.3f}",
                            test_loss=f"{test_metrics['loss']:.3f}",
                        )
                        progress_write(
                            epoch_progress,
                            (
                                f"[centralized][seed={seed}][fold={fold_idx}][epoch={epoch}/{epochs}] "
                                f"loss tr/va/te={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f}/{test_metrics['loss']:.4f} "
                                f"acc tr/va/te={train_metrics['acc']:.4f}/{val_metrics['acc']:.4f}/{test_metrics['acc']:.4f}"
                            ),
                        )

                        if val_metrics["auprc"] > best_val:
                            best_val = val_metrics["auprc"]
                            best_epoch = epoch
                            best_state = copy.deepcopy(model.state_dict())
                            epochs_no_improve = 0
                            torch.save(best_state, best_checkpoint_path)
                        else:
                            if epoch >= min_epochs:
                                epochs_no_improve += 1
                        if epoch >= min_epochs and epochs_no_improve >= patience:
                            break
                finally:
                    epoch_progress.close()

                model.load_state_dict(best_state)
                torch.save(model.state_dict(), final_checkpoint_path)
                test_metrics = _evaluate(model, test_loader, device, criterion)
                result = {
                    "seed": seed,
                    "fold": fold_idx,
                    "best_epoch": best_epoch,
                    "best_val_auprc": best_val,
                    "test": test_metrics,
                    "history": history,
                    "resumed": False,
                }
                all_results.append(result)
                result_checkpoint_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            finally:
                _shutdown_loader(train_loader)
                _shutdown_loader(val_loader)
                _shutdown_loader(test_loader)
                del train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
                if model is not None:
                    del model
                if optimizer is not None:
                    del optimizer
                if scheduler is not None:
                    del scheduler
                if criterion is not None:
                    del criterion
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    raw_results_path = experiment_dir / "raw_results.json"
    raw_results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    summary_rows = []
    for result in all_results:
        summary_rows.append(
            {
                "seed": result["seed"],
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
                "best_val_auprc": result["best_val_auprc"],
                **result["test"],
            }
        )
    write_csv(experiment_dir / "summary.csv", summary_rows, list(summary_rows[0].keys()) if summary_rows else ["seed"])

    aggregate = mean_std_across_runs(summary_rows, ["loss", "acc", "f1", "auroc", "auprc", "sens", "spec"])
    metrics_payload = {
        "method": "centralized",
        "experiment": experiment,
        "mode": "multimodal_concat",
        "seeds": seeds,
        "fold_metrics": all_results,
        "aggregate": {"ad_traj": aggregate},
    }
    (experiment_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (experiment_dir / "config_used.yaml").write_text(
        yaml.safe_dump(
            {
                "config_path": str(config_path),
                "output_dir": str(experiment_dir),
                "device": str(device),
                "epochs": epochs,
                "batch_size": batch_size,
                "folds": folds,
                "num_workers": num_workers,
                "seeds": seeds,
                "lr": lr,
                "weight_decay": weight_decay,
                "mri_shape": list(mri_shape),
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "patch_size": model_cfg.get("patch_size", [16, 16, 16]),
                "mri_dim": int(model_cfg.get("mri_dim", 768)),
                "token_dim": int(model_cfg.get("token_dim", 64)),
                "vit_layers": int(model_cfg.get("vit_layers", 12)),
                "vit_heads": int(model_cfg.get("vit_heads", 12)),
                "tab_layers": int(model_cfg.get("tab_layers", 3)),
                "tab_heads": int(model_cfg.get("tab_heads", 4)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {"experiment_dir": experiment_dir, "metrics": metrics_payload}
