"""Paper-aligned multimodal backbones for centralized and federated runs."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import ViT


class ViTEncoder(nn.Module):
    def __init__(
        self,
        *,
        img_size: Sequence[int] = (96, 96, 96),
        patch_size: Sequence[int] = (16, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.vit = ViT(
            in_channels=1,
            img_size=tuple(int(v) for v in img_size),
            patch_size=tuple(int(v) for v in patch_size),
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            classification=False,
            spatial_dims=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.vit(x.float())
        return tokens


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, token_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, token_dim))
        self.bias = nn.Parameter(torch.empty(num_features, token_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.float().unsqueeze(-1)
        return values * self.weight + self.bias


class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_features: int,
        token_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(seq)
        return self.norm(encoded)


class PaperSharedEncoders(nn.Module):
    """Shared ViT + FT-Transformer encoders with projected modality embeddings."""

    def __init__(
        self,
        *,
        num_features: int,
        img_size: Sequence[int] = (96, 96, 96),
        patch_size: Sequence[int] = (16, 16, 16),
        mri_dim: int = 768,
        token_dim: int = 64,
        projection_dim: int = 256,
        vit_layers: int = 12,
        vit_heads: int = 12,
        tab_layers: int = 3,
        tab_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mri_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=mri_dim,
            mlp_dim=mri_dim * 4,
            num_layers=vit_layers,
            num_heads=vit_heads,
            dropout_rate=dropout,
        )
        self.tab_encoder = FTTransformer(
            num_features=num_features,
            token_dim=token_dim,
            num_layers=tab_layers,
            num_heads=tab_heads,
            dropout=dropout,
        )
        self.mri_projection = nn.Sequential(
            nn.LayerNorm(mri_dim),
            nn.Linear(mri_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tab_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mri_missing = nn.Parameter(torch.zeros(1, projection_dim))
        self.tab_missing = nn.Parameter(torch.zeros(1, projection_dim))
        self.projection_dim = int(projection_dim)

    def encode(
        self,
        mri: torch.Tensor,
        tabular: torch.Tensor,
        mri_mask: torch.Tensor,
        tab_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mri_tokens = self.mri_encoder(mri)
        tab_tokens = self.tab_encoder(tabular)
        mri_embedding = self.mri_projection(mri_tokens[:, 0])
        tab_embedding = self.tab_projection(tab_tokens[:, 0])
        mri_mask = mri_mask.view(-1, 1).float()
        tab_mask = tab_mask.view(-1, 1).float()
        mri_embedding = mri_mask * mri_embedding + (1.0 - mri_mask) * self.mri_missing.expand(mri_embedding.shape[0], -1)
        tab_embedding = tab_mask * tab_embedding + (1.0 - tab_mask) * self.tab_missing.expand(tab_embedding.shape[0], -1)
        return mri_embedding, tab_embedding


class PaperFedAvgModel(nn.Module):
    """ViT + FT-Transformer baseline with simple concatenation fusion."""

    def __init__(
        self,
        *,
        num_features: int,
        img_size: Sequence[int],
        patch_size: Sequence[int] = (16, 16, 16),
        mri_dim: int = 768,
        token_dim: int = 64,
        projection_dim: int = 256,
        attn_dim: int = 256,
        vit_layers: int = 12,
        vit_heads: int = 12,
        tab_layers: int = 3,
        tab_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.shared = PaperSharedEncoders(
            num_features=num_features,
            img_size=img_size,
            patch_size=patch_size,
            mri_dim=mri_dim,
            token_dim=token_dim,
            projection_dim=projection_dim,
            vit_layers=vit_layers,
            vit_heads=vit_heads,
            tab_layers=tab_layers,
            tab_heads=tab_heads,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(projection_dim * 2 + 2, attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, num_classes),
        )

    def forward(
        self,
        mri: torch.Tensor,
        tabular: torch.Tensor,
        mri_mask: torch.Tensor,
        tab_mask: torch.Tensor,
    ) -> torch.Tensor:
        mri_embedding, tab_embedding = self.shared.encode(mri, tabular, mri_mask, tab_mask)
        fused = torch.cat([mri_embedding, tab_embedding, mri_mask.unsqueeze(1), tab_mask.unsqueeze(1)], dim=-1)
        return self.classifier(self.fusion(fused))


class PaperFedGateModel(nn.Module):
    """ViT + FT-Transformer encoders with a local modality gating network."""

    def __init__(
        self,
        *,
        num_features: int,
        img_size: Sequence[int],
        patch_size: Sequence[int] = (16, 16, 16),
        mri_dim: int = 768,
        token_dim: int = 64,
        projection_dim: int = 256,
        hidden_dim: int = 128,
        vit_layers: int = 12,
        vit_heads: int = 12,
        tab_layers: int = 3,
        tab_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.shared = PaperSharedEncoders(
            num_features=num_features,
            img_size=img_size,
            patch_size=patch_size,
            mri_dim=mri_dim,
            token_dim=token_dim,
            projection_dim=projection_dim,
            vit_layers=vit_layers,
            vit_heads=vit_heads,
            tab_layers=tab_layers,
            tab_heads=tab_heads,
            dropout=dropout,
        )
        self.gating = nn.Sequential(
            nn.Linear(projection_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        mri: torch.Tensor,
        tabular: torch.Tensor,
        mri_mask: torch.Tensor,
        tab_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mri_embedding, tab_embedding = self.shared.encode(mri, tabular, mri_mask, tab_mask)
        gate_inputs = torch.cat([mri_embedding, tab_embedding, mri_mask.unsqueeze(1), tab_mask.unsqueeze(1)], dim=1)
        gate_logits = self.gating(gate_inputs)
        availability_mask = torch.stack([mri_mask, tab_mask], dim=1)
        gate_logits = gate_logits.masked_fill(availability_mask <= 0.0, -1e9)
        gate_weights = torch.softmax(gate_logits, dim=1)

        denom = availability_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        fallback = (mri_embedding * mri_mask.unsqueeze(1) + tab_embedding * tab_mask.unsqueeze(1)) / denom
        fused = gate_weights[:, :1] * mri_embedding + gate_weights[:, 1:] * tab_embedding
        fused = torch.where((availability_mask.sum(dim=1, keepdim=True) > 0.0), fused, fallback)
        return self.classifier(fused), gate_weights


def extract_shared_encoder_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items() if name.startswith("shared.")}


def load_shared_encoder_state(model: nn.Module, shared_state: Dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    current.update(shared_state)
    model.load_state_dict(current, strict=False)


def extract_classifier_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items() if name.startswith("classifier.")}


def load_classifier_state(model: nn.Module, classifier_state: Dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    current.update(classifier_state)
    model.load_state_dict(current, strict=False)


def average_named_tensors(weighted_states: Iterable[tuple[float, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    states = list(weighted_states)
    if not states:
        raise ValueError("Cannot average an empty collection of state dicts.")
    total_weight = float(sum(weight for weight, _ in states))
    if total_weight <= 0.0:
        raise ValueError("Total aggregation weight must be > 0.")

    keys = list(states[0][1].keys())
    aggregated: Dict[str, torch.Tensor] = {}
    for key in keys:
        accumulator = None
        for weight, state in states:
            tensor = state[key].float() * float(weight)
            accumulator = tensor if accumulator is None else accumulator + tensor
        aggregated[key] = (accumulator / total_weight).type_as(states[0][1][key])
    return aggregated


def extract_numpy_summary(state: Dict[str, torch.Tensor]) -> Dict[str, list[int]]:
    return {key: list(np.asarray(value.shape, dtype=np.int64)) for key, value in state.items()}
