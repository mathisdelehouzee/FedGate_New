"""Local federated simulation for FedAvg and FedGate benchmarks."""

from __future__ import annotations

import copy
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from fedgate_final.data.cbms import (
    FEATURES,
    RowData,
    ScenarioDataset,
    TabularScaler,
    build_mri_transforms,
    load_cbms_csv,
    load_scenario_seed_spec,
)
from fedgate_final.models.paper_multimodal import (
    PaperFedAvgModel,
    PaperFedGateModel,
    average_named_tensors,
    extract_shared_encoder_state,
)
from fedgate_final.training.metrics import (
    METRIC_NAMES,
    compute_binary_metrics,
    flatten_client_metrics,
    mean_std_across_runs,
    summarize_client_metrics,
    tune_binary_threshold,
    weighted_average_metrics,
    write_csv,
)
from fedgate_final.training.progress import make_progress, progress_write
from fedgate_final.training.runtime import apply_runtime_settings, resolve_mixed_precision_dtype


@dataclass
class ClientDataBundle:
    client_id: int
    client_type: str
    train_loader: DataLoader
    val_loader: DataLoader | None
    test_loader: DataLoader
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int


@dataclass
class SeedDataBundle:
    scenario_name: str
    scenario_description: str
    seed: int
    client_bundles: Dict[int, ClientDataBundle]
    canonical_test_loader: DataLoader
    mri_shape: Sequence[int]


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


def _rows_from_indices(rows: List[RowData], indices: List[int]) -> List[RowData]:
    return [rows[int(index)] for index in indices]


def load_seed_data(
    config_path: Path,
    seed: int,
    batch_size: int,
    num_workers: int,
    max_samples_per_client: int = 0,
) -> tuple[dict[str, Any], SeedDataBundle]:
    cfg, spec = load_scenario_seed_spec(config_path, seed)
    data_cfg = dict(cfg.get("data", {}))
    data_csv = _resolve_path(config_path.parent, str(data_cfg["csv"]))
    data_root = _resolve_path(config_path.parent, str(data_cfg["data_root"]))
    rows = load_cbms_csv(data_csv, data_root=data_root)
    mri_cache_cfg = dict(data_cfg.get("mri_cache", {}))
    mri_cache_enabled = bool(mri_cache_cfg.get("enabled", True))
    mri_cache_max_items = int(mri_cache_cfg.get("max_items", 1024))
    mri_cache_dir = (
        _resolve_path(config_path.parent, str(mri_cache_cfg["dir"]))
        if mri_cache_cfg.get("dir")
        else None
    )
    prefetch_factor = int(cfg.get("federated", {}).get("prefetch_factor", 4))

    mri_shape = _to_tuple3(data_cfg.get("mri_shape", (128, 128, 128)))
    transforms = build_mri_transforms(mri_shape)
    client_bundles: Dict[int, ClientDataBundle] = {}
    pin_memory = torch.cuda.is_available()

    for client_id, train_indices in spec.client_train_indices.items():
        train_rows = _rows_from_indices(rows, train_indices)
        val_rows = _rows_from_indices(rows, spec.client_val_indices.get(client_id, []))
        if max_samples_per_client > 0:
            train_rows = train_rows[:max_samples_per_client]
            if val_rows:
                val_rows = val_rows[: max(1, min(len(val_rows), max_samples_per_client))]
        scaler = TabularScaler()
        scaler.fit(train_rows if train_rows else _rows_from_indices(rows, spec.client_all_indices[client_id]))
        client_type = spec.client_types[client_id]

        train_ds = ScenarioDataset(
            rows=train_rows,
            tabular_scaler=scaler,
            mri_transforms=transforms,
            client_id=client_id,
            client_type=client_type,
            split_name="train",
            scenario_cfg=spec.scenario_cfg,
            seed=seed,
            mri_shape=mri_shape,
            mri_cache_enabled=mri_cache_enabled,
            mri_cache_max_items=mri_cache_max_items,
            mri_cache_dir=mri_cache_dir,
        )
        val_ds = (
            ScenarioDataset(
                rows=val_rows,
                tabular_scaler=scaler,
                mri_transforms=transforms,
                client_id=client_id,
                client_type=client_type,
                split_name="val",
                scenario_cfg=spec.scenario_cfg,
                seed=seed,
                mri_shape=mri_shape,
                mri_cache_enabled=mri_cache_enabled,
                mri_cache_max_items=mri_cache_max_items,
                mri_cache_dir=mri_cache_dir,
            )
            if val_rows
            else None
        )
        test_ds = ScenarioDataset(
            rows=_rows_from_indices(rows, spec.global_test_indices),
            tabular_scaler=scaler,
            mri_transforms=transforms,
            client_id=client_id,
            client_type=client_type,
            split_name="test",
            scenario_cfg=spec.scenario_cfg,
            seed=seed,
            mri_shape=mri_shape,
            mri_cache_enabled=mri_cache_enabled,
            mri_cache_max_items=mri_cache_max_items,
            mri_cache_dir=mri_cache_dir,
        )
        client_bundles[client_id] = ClientDataBundle(
            client_id=client_id,
            client_type=client_type,
            train_loader=DataLoader(
                train_ds,
                **_loader_kwargs(
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_factor=prefetch_factor,
                ),
            ),
            val_loader=(
                DataLoader(
                    val_ds,
                    **_loader_kwargs(
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        prefetch_factor=prefetch_factor,
                    ),
                )
                if val_ds is not None
                else None
            ),
            test_loader=DataLoader(
                test_ds,
                **_loader_kwargs(
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_factor=prefetch_factor,
                ),
            ),
            num_train_samples=len(train_ds),
            num_val_samples=len(val_ds) if val_ds is not None else 0,
            num_test_samples=len(test_ds),
        )

    canonical_scaler = TabularScaler()
    canonical_train_indices: List[int] = []
    for indices in spec.client_train_indices.values():
        canonical_train_indices.extend(indices)
    canonical_train_rows = _rows_from_indices(rows, canonical_train_indices)
    canonical_scaler.fit(canonical_train_rows)
    canonical_test_loader = DataLoader(
        ScenarioDataset(
            rows=_rows_from_indices(rows, spec.global_test_indices),
            tabular_scaler=canonical_scaler,
            mri_transforms=transforms,
            client_id=-1,
            client_type="both_aligned",
            split_name="test",
            scenario_cfg={"apply_to_test": False},
            seed=seed,
            mri_shape=mri_shape,
            mri_cache_enabled=mri_cache_enabled,
            mri_cache_max_items=mri_cache_max_items,
            mri_cache_dir=mri_cache_dir,
        ),
        **_loader_kwargs(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        ),
    )
    return cfg, SeedDataBundle(
        scenario_name=spec.scenario_name,
        scenario_description=spec.scenario_description,
        seed=seed,
        client_bundles=client_bundles,
        canonical_test_loader=canonical_test_loader,
        mri_shape=mri_shape,
    )


def _forward(model: nn.Module, batch: Dict[str, torch.Tensor], method: str) -> torch.Tensor:
    if method == "fedgate":
        logits, _ = model(batch["mri"], batch["tabular"], batch["mri_mask"], batch["tab_mask"])
        return logits
    return model(batch["mri"], batch["tabular"], batch["mri_mask"], batch["tab_mask"])


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


def _autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _collect_predictions(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    amp_dtype: torch.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = torch.zeros((), device=device)
    total_examples = 0
    y_true_chunks: List[torch.Tensor] = []
    y_prob_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            with _autocast_context(device, amp_dtype):
                logits = _forward(model, batch, method)
                loss = criterion(logits, batch["label"])
            batch_size = int(batch["label"].size(0))
            y_true_chunks.append(batch["label"].detach())
            y_prob_chunks.append(torch.softmax(logits, dim=1)[:, 1].detach().float())
            total_examples += batch_size
            total_loss = total_loss + loss.detach() * batch_size
    avg_loss = float((total_loss / max(1, total_examples)).item())
    y_true = torch.cat(y_true_chunks).cpu().numpy() if y_true_chunks else np.asarray([], dtype=np.int64)
    y_prob = torch.cat(y_prob_chunks).cpu().numpy() if y_prob_chunks else np.asarray([], dtype=np.float32)
    return y_true, y_prob, avg_loss


def _resolve_threshold(
    evaluation_cfg: Dict[str, Any],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    threshold_cfg = dict(evaluation_cfg.get("threshold", {}))
    mode = str(threshold_cfg.get("mode", "fixed")).strip().lower()
    if mode == "tuned_on_val":
        return tune_binary_threshold(
            y_true,
            y_prob,
            metric=str(threshold_cfg.get("metric", "f1")),
            min_value=float(threshold_cfg.get("min", 0.05)),
            max_value=float(threshold_cfg.get("max", 0.95)),
            num_points=int(threshold_cfg.get("num_points", 181)),
        )
    return float(threshold_cfg.get("value", 0.5))


def _nan_metrics(*, threshold: float) -> Dict[str, float]:
    metrics = {name: float("nan") for name in METRIC_NAMES}
    metrics["threshold"] = float(threshold)
    return metrics


def train_local_model(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    local_epochs: int,
    method: str,
    l1_gate: float = 0.0,
    amp_dtype: torch.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    model = model.to(device)
    if not hasattr(loader.dataset, "rows"):
        raise ValueError("Expected a scenario dataset with a `rows` attribute.")
    labels = [int(row.label) for row in loader.dataset.rows]
    class_weights = _compute_class_weights(labels, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=device.type == "cuda" and amp_dtype == torch.float16,
    )

    total_loss = torch.zeros((), device=device)
    total_examples = 0
    y_true_chunks: List[torch.Tensor] = []
    y_prob_chunks: List[torch.Tensor] = []
    for _ in range(local_epochs):
        model.train()
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_dtype):
                logits = _forward(model, batch, method)
                loss = criterion(logits, batch["label"])
                if method == "fedgate" and l1_gate > 0.0 and hasattr(model, "gating"):
                    gate_penalty = 0.0
                    for name, param in model.named_parameters():
                        if name.startswith("gating."):
                            gate_penalty = gate_penalty + param.abs().mean()
                    loss = loss + float(l1_gate) * gate_penalty
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_size = int(batch["label"].size(0))
            y_true_chunks.append(batch["label"].detach())
            y_prob_chunks.append(torch.softmax(logits, dim=1)[:, 1].detach().float())
            total_examples += batch_size
            total_loss = total_loss + loss.detach() * batch_size
    avg_loss = float((total_loss / max(1, total_examples)).item())
    y_true = torch.cat(y_true_chunks).cpu().numpy() if y_true_chunks else np.asarray([], dtype=np.int64)
    y_prob = torch.cat(y_prob_chunks).cpu().numpy() if y_prob_chunks else np.asarray([], dtype=np.float32)
    return y_true, y_prob, avg_loss


@torch.no_grad()
def evaluate_model(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    amp_dtype: torch.dtype | None = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true, y_prob, avg_loss = _collect_predictions(
        model=model,
        loader=loader,
        device=device,
        method=method,
        amp_dtype=amp_dtype,
    )
    return compute_binary_metrics(y_true, y_prob, loss=avg_loss, threshold=threshold)


def _state_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _aggregate_full_models(weighted_models: List[tuple[int, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    return average_named_tensors([(float(weight), state) for weight, state in weighted_models])


def _shared_parameter_scope(method: str) -> str:
    return "full_model" if method == "fedavg" else "encoders_only"


def _make_model(method: str, *, mri_shape: Sequence[int], hidden_dim: int, dropout: float) -> nn.Module:
    return _make_model_with_config(
        method=method,
        mri_shape=mri_shape,
        hidden_dim=hidden_dim,
        dropout=dropout,
        model_cfg={},
    )


def _make_model_with_config(
    method: str,
    *,
    mri_shape: Sequence[int],
    hidden_dim: int,
    dropout: float,
    model_cfg: Dict[str, Any],
) -> nn.Module:
    patch_size = tuple(int(v) for v in model_cfg.get("patch_size", (16, 16, 16)))
    mri_dim = int(model_cfg.get("mri_dim", 768))
    token_dim = int(model_cfg.get("token_dim", 64))
    vit_layers = int(model_cfg.get("vit_layers", 12))
    vit_heads = int(model_cfg.get("vit_heads", 12))
    tab_layers = int(model_cfg.get("tab_layers", 3))
    tab_heads = int(model_cfg.get("tab_heads", 4))
    if method == "fedavg":
        return PaperFedAvgModel(
            num_features=len(FEATURES),
            img_size=mri_shape,
            patch_size=patch_size,
            mri_dim=mri_dim,
            token_dim=token_dim,
            projection_dim=hidden_dim,
            attn_dim=hidden_dim,
            dropout=dropout,
            vit_layers=vit_layers,
            vit_heads=vit_heads,
            tab_layers=tab_layers,
            tab_heads=tab_heads,
        )
    if method == "fedgate":
        return PaperFedGateModel(
            num_features=len(FEATURES),
            img_size=mri_shape,
            patch_size=patch_size,
            mri_dim=mri_dim,
            token_dim=token_dim,
            projection_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            vit_layers=vit_layers,
            vit_heads=vit_heads,
            tab_layers=tab_layers,
            tab_heads=tab_heads,
        )
    raise ValueError(f"Unsupported method: {method}")


def _select_clients(client_ids: List[int], fraction_fit: float, seed: int, server_round: int) -> List[int]:
    if fraction_fit >= 1.0:
        return client_ids
    rng = random.Random(seed * 1000 + server_round)
    num_selected = max(1, int(round(len(client_ids) * fraction_fit)))
    return sorted(rng.sample(client_ids, num_selected))


def run_seed(
    *,
    method: str,
    config_path: Path,
    seed: int,
    output_dir: Path,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    fraction_fit: float,
    device: torch.device,
    num_workers: int,
    max_samples_per_client: int,
    hidden_dim: int,
    dropout: float,
    l1_gate: float,
    amp_dtype: torch.dtype | None,
    evaluation_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    set_seed(seed)
    cfg, data_bundle = load_seed_data(
        config_path,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples_per_client=max_samples_per_client,
    )
    client_ids = sorted(data_bundle.client_bundles.keys())
    scenario_name = data_bundle.scenario_name
    scenario_description = data_bundle.scenario_description
    global_history: List[Dict[str, Any]] = []
    per_round_rows: List[Dict[str, Any]] = []

    model_global = _make_model_with_config(
        method,
        mri_shape=data_bundle.mri_shape,
        hidden_dim=hidden_dim,
        dropout=dropout,
        model_cfg=model_cfg,
    )
    global_state = _state_to_cpu(model_global.state_dict())
    initial_personal_state = _state_to_cpu(
        _make_model_with_config(
            method,
            mri_shape=data_bundle.mri_shape,
            hidden_dim=hidden_dim,
            dropout=dropout,
            model_cfg=model_cfg,
        ).state_dict()
    )
    client_personal_states = {
        client_id: copy.deepcopy(initial_personal_state)
        for client_id in client_ids
    }

    best_metric = float("-inf")
    best_round = 0
    best_global_state = copy.deepcopy(global_state)
    best_client_states = copy.deepcopy(client_personal_states)
    val_every_rounds = max(1, int(evaluation_cfg.get("val_every_rounds", 1)))
    test_every_rounds = max(1, int(evaluation_cfg.get("test_every_rounds", 1)))
    last_threshold = float(dict(evaluation_cfg.get("threshold", {})).get("value", 0.5))
    last_round_val = _nan_metrics(threshold=last_threshold)
    last_round_test = _nan_metrics(threshold=last_threshold)

    round_progress = make_progress(
        range(1, rounds + 1),
        total=rounds,
        desc=f"{method} seed={seed}",
        leave=True,
    )
    for server_round in round_progress:
        selected_clients = _select_clients(client_ids, fraction_fit=fraction_fit, seed=seed, server_round=server_round)
        train_items: List[tuple[int, Dict[str, float]]] = []
        cached_train_outputs: Dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        aggregated_states: List[tuple[int, Dict[str, torch.Tensor]]] = []

        for client_id in selected_clients:
            client_bundle = data_bundle.client_bundles[client_id]
            local_model = _make_model_with_config(
                method,
                mri_shape=data_bundle.mri_shape,
                hidden_dim=hidden_dim,
                dropout=dropout,
                model_cfg=model_cfg,
            )
            if method == "fedavg":
                local_model.load_state_dict(global_state, strict=True)
            else:
                local_model.load_state_dict(client_personal_states[client_id], strict=True)
                shared_state = extract_shared_encoder_state(model_global)
                current_state = local_model.state_dict()
                current_state.update(shared_state)
                local_model.load_state_dict(current_state, strict=False)

            train_y_true, train_y_prob, train_avg_loss = train_local_model(
                model=local_model,
                loader=client_bundle.train_loader,
                device=device,
                lr=lr,
                weight_decay=weight_decay,
                local_epochs=local_epochs,
                method=method,
                l1_gate=l1_gate,
                amp_dtype=amp_dtype,
            )
            cached_train_outputs[client_id] = (train_y_true, train_y_prob, train_avg_loss)

            if method == "fedavg":
                aggregated_states.append((client_bundle.num_train_samples, _state_to_cpu(local_model.state_dict())))
            else:
                client_personal_states[client_id] = _state_to_cpu(local_model.state_dict())
                aggregated_states.append((client_bundle.num_train_samples, extract_shared_encoder_state(local_model)))

        if method == "fedavg":
            global_state = _aggregate_full_models(aggregated_states)
            model_global.load_state_dict(global_state, strict=True)
        else:
            shared_state = average_named_tensors([(float(weight), state) for weight, state in aggregated_states])
            current_state = model_global.state_dict()
            current_state.update(shared_state)
            model_global.load_state_dict(current_state, strict=False)

        if method == "fedgate":
            # Refresh every client with the new global encoders before validation.
            for client_id in client_ids:
                refreshed = _make_model_with_config(
                    method,
                    mri_shape=data_bundle.mri_shape,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    model_cfg=model_cfg,
                )
                refreshed.load_state_dict(client_personal_states[client_id], strict=True)
                current_state = refreshed.state_dict()
                current_state.update(extract_shared_encoder_state(model_global))
                refreshed.load_state_dict(current_state, strict=False)
                client_personal_states[client_id] = _state_to_cpu(refreshed.state_dict())

        run_val = (server_round == 1) or (server_round % val_every_rounds == 0) or (server_round == rounds)
        run_test = (server_round == 1) or (server_round % test_every_rounds == 0) or (server_round == rounds)

        client_val_metrics: Dict[str, Dict[str, Any]] = {}
        round_threshold = last_threshold
        if run_val:
            round_val_items: List[tuple[int, Dict[str, float]]] = []
            eval_model = _make_model_with_config(
                method,
                mri_shape=data_bundle.mri_shape,
                hidden_dim=hidden_dim,
                dropout=dropout,
                model_cfg=model_cfg,
            )
            val_truth_parts: List[np.ndarray] = []
            val_prob_parts: List[np.ndarray] = []
            cached_val_outputs: Dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
            for client_id in client_ids:
                client_bundle = data_bundle.client_bundles[client_id]
                if method == "fedavg":
                    eval_model.load_state_dict(global_state, strict=True)
                else:
                    eval_model.load_state_dict(client_personal_states[client_id], strict=True)
                if client_bundle.val_loader is not None:
                    y_true, y_prob, avg_loss = _collect_predictions(
                        model=eval_model,
                        loader=client_bundle.val_loader,
                        device=device,
                        method=method,
                        amp_dtype=amp_dtype,
                    )
                    cached_val_outputs[client_id] = (y_true, y_prob, avg_loss)
                    if y_true.size > 0:
                        val_truth_parts.append(y_true)
                        val_prob_parts.append(y_prob)

            round_threshold = _resolve_threshold(
                evaluation_cfg,
                np.concatenate(val_truth_parts) if val_truth_parts else np.asarray([], dtype=np.int64),
                np.concatenate(val_prob_parts) if val_prob_parts else np.asarray([], dtype=np.float32),
            )
            last_threshold = round_threshold

            for client_id in client_ids:
                client_bundle = data_bundle.client_bundles[client_id]
                if client_bundle.val_loader is not None:
                    y_true, y_prob, avg_loss = cached_val_outputs[client_id]
                    val_metrics = compute_binary_metrics(y_true, y_prob, loss=avg_loss, threshold=round_threshold)
                    round_val_items.append((client_bundle.num_val_samples, val_metrics))
                    client_val_metrics[f"client_{client_id}"] = {
                        "split": "val",
                        "client_type": client_bundle.client_type,
                        "num_samples": client_bundle.num_val_samples,
                        **val_metrics,
                    }
            round_val = weighted_average_metrics(round_val_items)
            last_round_val = round_val
        else:
            round_val = dict(last_round_val)
            round_val["threshold"] = round_threshold

        for client_id in selected_clients:
            client_bundle = data_bundle.client_bundles[client_id]
            train_y_true, train_y_prob, train_avg_loss = cached_train_outputs[client_id]
            train_metrics = compute_binary_metrics(
                train_y_true,
                train_y_prob,
                loss=train_avg_loss,
                threshold=round_threshold,
            )
            train_items.append((client_bundle.num_train_samples, train_metrics))

        round_train = weighted_average_metrics(train_items)

        if run_test:
            round_test_items: List[tuple[int, Dict[str, float]]] = []
            test_model = _make_model_with_config(
                method,
                mri_shape=data_bundle.mri_shape,
                hidden_dim=hidden_dim,
                dropout=dropout,
                model_cfg=model_cfg,
            )
            for client_id in client_ids:
                client_bundle = data_bundle.client_bundles[client_id]
                if method == "fedavg":
                    test_model.load_state_dict(global_state, strict=True)
                else:
                    test_model.load_state_dict(client_personal_states[client_id], strict=True)
                test_metrics = evaluate_model(
                    model=test_model,
                    loader=client_bundle.test_loader,
                    device=device,
                    method=method,
                    amp_dtype=amp_dtype,
                    threshold=round_threshold,
                )
                round_test_items.append((client_bundle.num_test_samples, test_metrics))
            round_test = weighted_average_metrics(round_test_items)
            last_round_test = round_test
        else:
            round_test = dict(last_round_test)
            round_test["threshold"] = round_threshold
        history_record = {
            "round": server_round,
            "num_clients": len(selected_clients),
            "threshold": round_threshold,
            "val_evaluated": run_val,
            "test_evaluated": run_test,
            "train": round_train,
            "val": round_val,
            "test": round_test,
        }
        global_history.append(history_record)
        per_round_rows.append(
            {
                "round": server_round,
                "num_clients": len(selected_clients),
                "threshold": round_threshold,
                "val_evaluated": int(run_val),
                "test_evaluated": int(run_test),
                **{f"train_{key}": value for key, value in round_train.items()},
                **{f"val_{key}": value for key, value in round_val.items()},
                **{f"test_{key}": value for key, value in round_test.items()},
            }
        )
        round_progress.set_postfix(
            train_acc=f"{round_train['acc']:.3f}",
            val_acc=f"{round_val['acc']:.3f}",
            test_acc=f"{round_test['acc']:.3f}",
            train_loss=f"{round_train['loss']:.3f}",
            val_loss=f"{round_val['loss']:.3f}",
            test_loss=f"{round_test['loss']:.3f}",
        )
        progress_write(
            round_progress,
            (
                f"[{method}][seed={seed}][round={server_round}/{rounds}] "
                f"eval va/te={int(run_val)}/{int(run_test)} "
                f"loss tr/va/te={round_train['loss']:.4f}/{round_val['loss']:.4f}/{round_test['loss']:.4f} "
                f"acc tr/va/te={round_train['acc']:.4f}/{round_val['acc']:.4f}/{round_test['acc']:.4f} "
                f"auroc tr/va/te={round_train['auroc']:.4f}/{round_val['auroc']:.4f}/{round_test['auroc']:.4f} "
                f"auprc tr/va/te={round_train['auprc']:.4f}/{round_val['auprc']:.4f}/{round_test['auprc']:.4f}"
            ),
        )

        metric_value = float(round_val.get("auprc", float("nan")))
        if run_val and np.isfinite(metric_value) and metric_value > best_metric:
            best_metric = metric_value
            best_round = server_round
            best_global_state = _state_to_cpu(model_global.state_dict())
            best_client_states = copy.deepcopy(client_personal_states)
    round_progress.close()

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_global_state, checkpoints_dir / "best_global.pt")
    torch.save(model_global.state_dict(), checkpoints_dir / "final_global.pt")
    if method == "fedgate":
        torch.save(best_client_states, checkpoints_dir / "best_clients.pt")
        torch.save(client_personal_states, checkpoints_dir / "final_clients.pt")

    client_eval_metrics: Dict[str, Dict[str, Any]] = {}
    eval_model = _make_model_with_config(
        method,
        mri_shape=data_bundle.mri_shape,
        hidden_dim=hidden_dim,
        dropout=dropout,
        model_cfg=model_cfg,
    )
    final_val_truth_parts: List[np.ndarray] = []
    final_val_prob_parts: List[np.ndarray] = []
    cached_eval_outputs: Dict[int, tuple[np.ndarray, np.ndarray, float, str, int]] = {}
    for client_id in client_ids:
        client_bundle = data_bundle.client_bundles[client_id]
        if method == "fedavg":
            eval_model.load_state_dict(best_global_state, strict=True)
        else:
            eval_model.load_state_dict(best_client_states[client_id], strict=True)
        eval_loader = client_bundle.val_loader if client_bundle.val_loader is not None else client_bundle.test_loader
        eval_split = "val" if client_bundle.val_loader is not None else "global_test"
        eval_samples = client_bundle.num_val_samples if client_bundle.val_loader is not None else client_bundle.num_test_samples
        y_true, y_prob, avg_loss = _collect_predictions(
            model=eval_model,
            loader=eval_loader,
            device=device,
            method=method,
            amp_dtype=amp_dtype,
        )
        cached_eval_outputs[client_id] = (y_true, y_prob, avg_loss, eval_split, eval_samples)
        if eval_split == "val" and y_true.size > 0:
            final_val_truth_parts.append(y_true)
            final_val_prob_parts.append(y_prob)

    final_threshold = _resolve_threshold(
        evaluation_cfg,
        np.concatenate(final_val_truth_parts) if final_val_truth_parts else np.asarray([], dtype=np.int64),
        np.concatenate(final_val_prob_parts) if final_val_prob_parts else np.asarray([], dtype=np.float32),
    )

    for client_id in client_ids:
        client_bundle = data_bundle.client_bundles[client_id]
        y_true, y_prob, avg_loss, eval_split, eval_samples = cached_eval_outputs[client_id]
        eval_metrics = compute_binary_metrics(y_true, y_prob, loss=avg_loss, threshold=final_threshold)
        client_eval_metrics[f"client_{client_id}"] = {
            "split": eval_split,
            "client_type": client_bundle.client_type,
            "num_samples": eval_samples,
            **eval_metrics,
        }

    # Final held-out test metrics.
    client_test_metrics: Dict[str, Dict[str, Any]] = {}
    test_items: List[tuple[int, Dict[str, float]]] = []
    test_model = _make_model_with_config(
        method,
        mri_shape=data_bundle.mri_shape,
        hidden_dim=hidden_dim,
        dropout=dropout,
        model_cfg=model_cfg,
    )
    for client_id in client_ids:
        client_bundle = data_bundle.client_bundles[client_id]
        if method == "fedavg":
            test_model.load_state_dict(best_global_state, strict=True)
        else:
            test_model.load_state_dict(best_client_states[client_id], strict=True)
        test_metrics = evaluate_model(
            model=test_model,
            loader=client_bundle.test_loader,
            device=device,
            method=method,
            amp_dtype=amp_dtype,
            threshold=final_threshold,
        )
        client_test_metrics[f"client_{client_id}"] = {
            "split": "global_test",
            "client_type": client_bundle.client_type,
            "num_samples": client_bundle.num_test_samples,
            **test_metrics,
        }
        test_items.append((client_bundle.num_test_samples, test_metrics))

    final_test = weighted_average_metrics(test_items)
    client_summary = summarize_client_metrics(client_eval_metrics)

    canonical_test = None
    if method == "fedavg":
        eval_model = _make_model_with_config(
            method,
            mri_shape=data_bundle.mri_shape,
            hidden_dim=hidden_dim,
            dropout=dropout,
            model_cfg=model_cfg,
        )
        eval_model.load_state_dict(best_global_state, strict=True)
        canonical_test = evaluate_model(
            model=eval_model,
            loader=data_bundle.canonical_test_loader,
            device=device,
            method=method,
            amp_dtype=amp_dtype,
            threshold=final_threshold,
        )

    final_metrics_row = {
        "seed": seed,
        "best_round": best_round,
        "threshold": final_threshold,
        **final_test,
        "client_acc_std": client_summary["acc"]["std"],
        "client_auroc_std": client_summary["auroc"]["std"],
        "client_auprc_std": client_summary["auprc"]["std"],
        "client_acc_gap": client_summary["acc"]["fairness_gap"],
        "client_auroc_gap": client_summary["auroc"]["fairness_gap"],
        "client_auprc_gap": client_summary["auprc"]["fairness_gap"],
    }

    metrics_payload = {
        "method": method,
        "seed": seed,
        "scenario": {
            "name": scenario_name,
            "description": scenario_description,
        },
        "history": global_history,
        "best_round": best_round,
        "best_round_metric": "auprc",
        "decision_threshold": final_threshold,
        "final_test": final_test,
        "canonical_test": canonical_test,
        "client_eval_summary": client_summary,
        "shared_parameter_scope": _shared_parameter_scope(method),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (output_dir / "client_metrics.json").write_text(json.dumps(client_eval_metrics, indent=2), encoding="utf-8")
    (output_dir / "client_test_metrics.json").write_text(json.dumps(client_test_metrics, indent=2), encoding="utf-8")
    (output_dir / "final_metrics.json").write_text(json.dumps(final_metrics_row, indent=2), encoding="utf-8")
    write_csv(output_dir / "final_metrics.csv", [final_metrics_row], list(final_metrics_row.keys()))
    (output_dir / "per_round_metrics.json").write_text(json.dumps(global_history, indent=2), encoding="utf-8")
    write_csv(output_dir / "per_round_metrics.csv", per_round_rows, list(per_round_rows[0].keys()) if per_round_rows else ["round"])
    write_csv(
        output_dir / "client_metrics.csv",
        flatten_client_metrics(client_eval_metrics),
        list(flatten_client_metrics(client_eval_metrics)[0].keys()) if client_eval_metrics else ["client_id"],
    )
    write_csv(
        output_dir / "client_test_metrics.csv",
        flatten_client_metrics(client_test_metrics),
        list(flatten_client_metrics(client_test_metrics)[0].keys()) if client_test_metrics else ["client_id"],
    )

    return {
        "seed": seed,
        "scenario_name": scenario_name,
        "method": method,
        "final_metrics": final_metrics_row,
        "metrics_payload": metrics_payload,
    }


def run_experiment(
    *,
    method: str,
    config_path: Path,
    output_root: Path,
    seeds: List[int] | None = None,
    device_override: str = "",
    rounds_override: int = 0,
    local_epochs_override: int = 0,
    batch_size_override: int = 0,
    num_workers_override: int = -1,
    max_samples_per_client: int = 0,
) -> Dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    federated_cfg = dict(cfg.get("federated", {}))
    model_cfg = dict(cfg.get("model", {}))
    scenario_cfg = dict(cfg.get("scenario", {}))
    runtime_cfg = dict(cfg.get("runtime", {}))
    evaluation_cfg = dict(cfg.get("evaluation", {}))
    experiment = str(cfg.get("experiment", "fedgate_final"))
    scenario_name = str(scenario_cfg.get("name", "unknown"))
    run_seeds = seeds if seeds is not None else [int(seed) for seed in cfg.get("seeds", [7, 11, 17])]

    rounds = int(rounds_override or federated_cfg.get("num_server_rounds", 100))
    local_epochs = int(local_epochs_override or federated_cfg.get("local_epochs", 1))
    batch_size = int(batch_size_override or federated_cfg.get("batch_size", 4))
    num_workers = int(num_workers_override if num_workers_override >= 0 else federated_cfg.get("num_workers", 0))
    device = _pick_device(device_override or str(federated_cfg.get("device", "auto")))
    lr = float(federated_cfg.get("lr", 1e-4))
    weight_decay = float(federated_cfg.get("weight_decay", 1e-4))
    fraction_fit = float(federated_cfg.get("fraction_fit", 1.0))
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    dropout = float(model_cfg.get("dropout", 0.1))
    l1_gate = float(model_cfg.get("l1_gate", 0.0))
    apply_runtime_settings(runtime_cfg)
    amp_dtype = resolve_mixed_precision_dtype(runtime_cfg, device=device)

    experiment_dir = output_root / f"{experiment}_{method}_{scenario_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config_used = {
        "experiment": experiment,
        "method": method,
        "scenario": scenario_cfg,
        "config_path": str(config_path),
        "device": str(device),
        "seeds": run_seeds,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": lr,
        "weight_decay": weight_decay,
        "fraction_fit": fraction_fit,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "l1_gate": l1_gate,
        "patch_size": model_cfg.get("patch_size", [16, 16, 16]),
        "mri_dim": int(model_cfg.get("mri_dim", 768)),
        "token_dim": int(model_cfg.get("token_dim", 64)),
        "vit_layers": int(model_cfg.get("vit_layers", 12)),
        "vit_heads": int(model_cfg.get("vit_heads", 12)),
        "tab_layers": int(model_cfg.get("tab_layers", 3)),
        "tab_heads": int(model_cfg.get("tab_heads", 4)),
        "shared_parameter_scope": _shared_parameter_scope(method),
        "max_samples_per_client": max_samples_per_client,
        "mixed_precision": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "disabled",
    }
    (experiment_dir / "config_used.yaml").write_text(yaml.safe_dump(config_used, sort_keys=False), encoding="utf-8")

    seed_results = []
    for seed in run_seeds:
        seed_dir = experiment_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_results.append(
            run_seed(
                method=method,
                config_path=config_path,
                seed=seed,
                output_dir=seed_dir,
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                fraction_fit=fraction_fit,
                device=device,
                num_workers=num_workers,
                max_samples_per_client=max_samples_per_client,
                hidden_dim=hidden_dim,
                dropout=dropout,
                l1_gate=l1_gate,
                amp_dtype=amp_dtype,
                evaluation_cfg=evaluation_cfg,
                model_cfg=model_cfg,
            )
        )

    aggregate_rows = [result["final_metrics"] for result in seed_results]
    metric_keys = [
        "loss",
        "acc",
        "f1",
        "auroc",
        "auprc",
        "client_acc_std",
        "client_auroc_std",
        "client_auprc_std",
        "client_acc_gap",
        "client_auroc_gap",
        "client_auprc_gap",
    ]
    aggregate_payload = {
        "method": method,
        "scenario_name": scenario_name,
        "num_seeds": len(seed_results),
        "summary": mean_std_across_runs(aggregate_rows, metric_keys),
    }
    (experiment_dir / "aggregate_mean_std.json").write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")
    write_csv(experiment_dir / "summary.csv", aggregate_rows, list(aggregate_rows[0].keys()) if aggregate_rows else ["seed"])
    return {
        "experiment_dir": experiment_dir,
        "config_used": config_used,
        "aggregate": aggregate_payload,
    }
