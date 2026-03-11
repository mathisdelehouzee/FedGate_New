"""CBMS data and scenario utilities for benchmarkable centralized/FL runs."""

from __future__ import annotations

import csv
import hashlib
import os
import random
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import yaml
from monai.transforms import Compose, EnsureChannelFirst, EnsureType, LoadImage, Resize, ScaleIntensity
from torch.utils.data import Dataset

FEATURES = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "PTMARRY",
    "CATANIMSC",
    "TRAASCOR",
    "TRABSCOR",
    "DSPANFOR",
    "DSPANBAC",
    "BNTTOTAL",
    "VSWEIGHT",
    "BMI",
    "MH14ALCH",
    "MH16SMOK",
    "MH4CARD",
    "MH2NEURL",
]

CLIENT_TYPE_ORDER = ("both_aligned", "both_partial", "mri_only", "tab_only")
_MRI_CACHE: "OrderedDict[tuple[str, tuple[int, int, int]], torch.Tensor]" = OrderedDict()


@dataclass(frozen=True)
class RowData:
    index: int
    subject_id: str
    source: str
    label: int
    established_ad: int
    scan_path: Path
    features: List[Optional[float]]


@dataclass(frozen=True)
class ScenarioSeedSpec:
    scenario_name: str
    scenario_description: str
    seed: int
    global_test_indices: List[int]
    client_train_indices: Dict[int, List[int]]
    client_val_indices: Dict[int, List[int]]
    client_all_indices: Dict[int, List[int]]
    client_types: Dict[int, str]
    scenario_cfg: Dict[str, Any]


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.upper() == "NA":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def load_cbms_csv(
    csv_path: Path,
    data_root: Path,
    features: Sequence[str] = FEATURES,
) -> List[RowData]:
    rows: List[RowData] = []
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            scan_path = Path(row["scan_path"])
            if not scan_path.is_absolute():
                scan_path = data_root / scan_path
            rows.append(
                RowData(
                    index=idx,
                    subject_id=row["subject_id"],
                    source=row["source"],
                    label=int(row["label"]),
                    established_ad=int(row.get("established_ad", row["label"])),
                    scan_path=scan_path,
                    features=[_to_float(row.get(feature, "")) for feature in features],
                )
            )
    return rows


class TabularScaler:
    """Median imputation + z-score normalization fitted on train rows only."""

    def __init__(self) -> None:
        self.median: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, rows: Iterable[RowData]) -> None:
        values = [[np.nan if value is None else float(value) for value in row.features] for row in rows]
        data = np.asarray(values, dtype=np.float32)
        if data.size == 0:
            raise ValueError("Cannot fit TabularScaler on an empty row collection.")
        median = np.nanmedian(data, axis=0)
        imputed = np.where(np.isnan(data), median, data)
        mean = imputed.mean(axis=0)
        std = imputed.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.median = median
        self.mean = mean
        self.std = std

    def transform(self, values: Sequence[Optional[float]]) -> np.ndarray:
        if self.median is None or self.mean is None or self.std is None:
            raise ValueError("TabularScaler must be fitted before calling transform.")
        arr = np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float32)
        arr = np.where(np.isnan(arr), self.median, arr)
        arr = (arr - self.mean) / self.std
        return arr.astype(np.float32)


def build_mri_transforms(target_shape: Sequence[int]) -> Compose:
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            Resize(spatial_size=tuple(target_shape), mode="trilinear", align_corners=False),
            EnsureType(data_type="tensor"),
        ]
    )


def _get_cached_mri(
    scan_path: Path,
    mri_transforms: Compose,
    mri_shape: Sequence[int],
    *,
    cache_enabled: bool,
    cache_max_items: int,
    cache_dir: Path | None = None,
) -> torch.Tensor:
    key = (str(scan_path), tuple(int(v) for v in mri_shape))
    cache_path = mri_cache_path(scan_path, mri_shape, cache_dir) if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            tensor = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache_enabled:
            _MRI_CACHE[key] = tensor
            while len(_MRI_CACHE) > max(1, int(cache_max_items)):
                _MRI_CACHE.popitem(last=False)
        return tensor.clone()
    if cache_enabled and key in _MRI_CACHE:
        tensor = _MRI_CACHE.pop(key)
        _MRI_CACHE[key] = tensor
        return tensor.clone()

    tensor = mri_transforms(scan_path)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp.{os.getpid()}")
        torch.save(tensor.detach().cpu(), tmp_path)
        os.replace(tmp_path, cache_path)
    if cache_enabled:
        _MRI_CACHE[key] = tensor.detach().cpu()
        while len(_MRI_CACHE) > max(1, int(cache_max_items)):
            _MRI_CACHE.popitem(last=False)
        return _MRI_CACHE[key].clone()
    return tensor


def mri_cache_path(scan_path: Path, mri_shape: Sequence[int], cache_dir: Path) -> Path:
    digest = hashlib.sha1(f"{scan_path.resolve()}|{tuple(int(v) for v in mri_shape)}".encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.pt"


def cache_mri_volume(
    scan_path: Path,
    mri_transforms: Compose,
    mri_shape: Sequence[int],
    *,
    cache_dir: Path,
    cache_enabled: bool = True,
    cache_max_items: int = 0,
) -> Path:
    _get_cached_mri(
        scan_path,
        mri_transforms,
        mri_shape,
        cache_enabled=cache_enabled,
        cache_max_items=cache_max_items,
        cache_dir=cache_dir,
    )
    return mri_cache_path(scan_path, mri_shape, cache_dir)


def load_scenario_seed_spec(config_path: Path, seed: int) -> tuple[dict[str, Any], ScenarioSeedSpec]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Scenario config must be a mapping: {config_path}")

    artifacts_root = _resolve_path(config_path.parent, str(cfg["paths"]["artifacts_root"]))
    manifest_path = artifacts_root / "splits" / "splits_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing scenario manifest: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    seed_payload = manifest.get("seeds", {}).get(str(seed))
    if seed_payload is None:
        raise KeyError(f"Seed {seed} is not present in {manifest_path}")

    scenario_cfg = dict(cfg.get("scenario", {}))
    num_clients = int(cfg.get("partition", {}).get("num_clients", cfg.get("federated", {}).get("num_clients", 10)))
    client_types = assign_client_types(num_clients=num_clients, scenario_cfg=scenario_cfg, seed=seed)

    spec = ScenarioSeedSpec(
        scenario_name=str(scenario_cfg.get("name", "unknown")),
        scenario_description=str(scenario_cfg.get("description", "")),
        seed=int(seed),
        global_test_indices=[int(index) for index in manifest["global_split"]["test_indices"]],
        client_train_indices={
            int(cid): [int(index) for index in payload["train_indices"]]
            for cid, payload in seed_payload["client_splits"].items()
        },
        client_val_indices={
            int(cid): [int(index) for index in payload["val_indices"]]
            for cid, payload in seed_payload["client_splits"].items()
        },
        client_all_indices={
            int(cid): [int(index) for index in indices]
            for cid, indices in seed_payload["client_partitions"].items()
        },
        client_types=client_types,
        scenario_cfg=scenario_cfg,
    )
    return cfg, spec


def assign_client_types(num_clients: int, scenario_cfg: Dict[str, Any], seed: int) -> Dict[int, str]:
    counts = dict(scenario_cfg.get("client_type_counts", {}))
    client_ids = list(range(num_clients))
    rng = random.Random(seed)
    rng.shuffle(client_ids)

    assigned: Dict[int, str] = {}
    cursor = 0
    for client_type in CLIENT_TYPE_ORDER:
        count = int(counts.get(client_type, 0))
        for client_id in client_ids[cursor : cursor + count]:
            assigned[client_id] = client_type
        cursor += count

    for client_id in client_ids[cursor:]:
        assigned[client_id] = "both_aligned"
    return assigned


def _stable_fraction(*parts: str) -> float:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def resolve_modality_presence(
    *,
    subject_id: str,
    client_id: int,
    client_type: str,
    split_name: str,
    scenario_cfg: Dict[str, Any],
    seed: int,
) -> tuple[bool, bool]:
    apply_to_val = bool(scenario_cfg.get("apply_to_val", True))
    apply_to_test = bool(scenario_cfg.get("apply_to_test", False))
    if split_name == "val" and not apply_to_val:
        return True, True
    if split_name == "test" and not apply_to_test:
        return True, True

    if client_type == "both_aligned":
        return True, True
    if client_type == "mri_only":
        return True, False
    if client_type == "tab_only":
        return False, True
    if client_type != "both_partial":
        return True, True

    partial_pairing_rate = float(scenario_cfg.get("partial_pairing_rate", 1.0))
    paired_score = _stable_fraction(subject_id, str(client_id), str(seed), "paired")
    if paired_score <= partial_pairing_rate:
        return True, True
    missing_score = _stable_fraction(subject_id, str(client_id), str(seed), "missing")
    if missing_score < 0.5:
        return True, False
    return False, True


class ScenarioDataset(Dataset):
    """MRI + tabular dataset with deterministic scenario masking."""

    def __init__(
        self,
        *,
        rows: List[RowData],
        tabular_scaler: TabularScaler,
        mri_transforms: Compose,
        client_id: int,
        client_type: str,
        split_name: str,
        scenario_cfg: Dict[str, Any],
        seed: int,
        mri_shape: Sequence[int],
        mri_cache_enabled: bool = False,
        mri_cache_max_items: int = 512,
        mri_cache_dir: Path | None = None,
    ) -> None:
        self.rows = rows
        self.tabular_scaler = tabular_scaler
        self.mri_transforms = mri_transforms
        self.client_id = int(client_id)
        self.client_type = client_type
        self.split_name = split_name
        self.scenario_cfg = scenario_cfg
        self.seed = int(seed)
        self.mri_shape = tuple(int(v) for v in mri_shape)
        self.mri_cache_enabled = bool(mri_cache_enabled)
        self.mri_cache_max_items = int(mri_cache_max_items)
        self.mri_cache_dir = None if mri_cache_dir is None else Path(mri_cache_dir)
        self.zero_mri = torch.zeros((1, *tuple(int(v) for v in mri_shape)), dtype=torch.float32)
        self.zero_tab = torch.zeros((len(FEATURES),), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        mri_present, tab_present = resolve_modality_presence(
            subject_id=row.subject_id,
            client_id=self.client_id,
            client_type=self.client_type,
            split_name=self.split_name,
            scenario_cfg=self.scenario_cfg,
            seed=self.seed,
        )
        if mri_present:
            mri = _get_cached_mri(
                row.scan_path,
                self.mri_transforms,
                self.mri_shape,
                cache_enabled=self.mri_cache_enabled,
                cache_max_items=self.mri_cache_max_items,
                cache_dir=self.mri_cache_dir,
            )
        else:
            mri = self.zero_mri.clone()
        if tab_present:
            tabular = torch.from_numpy(self.tabular_scaler.transform(row.features))
        else:
            tabular = self.zero_tab.clone()
        return {
            "mri": mri,
            "tabular": tabular,
            "mri_mask": torch.tensor(float(mri_present), dtype=torch.float32),
            "tab_mask": torch.tensor(float(tab_present), dtype=torch.float32),
            "label": torch.tensor(row.label, dtype=torch.long),
            "established_ad": torch.tensor(row.established_ad, dtype=torch.long),
            "subject_id": row.subject_id,
            "source": row.source,
            "client_id": torch.tensor(self.client_id, dtype=torch.long),
        }
