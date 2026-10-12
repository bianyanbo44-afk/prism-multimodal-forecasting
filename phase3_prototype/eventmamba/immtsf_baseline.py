"""Protocol-aligned wrapper around the official IMM-TSF fusion modules.

The upstream MIT-licensed modules are imported from the pinned checkout rather
than copied.  This wrapper supplies our leakage-audited packed windows and makes
the text mask explicit so semantic ablations preserve event timestamps/counts.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Literal

import torch
from torch import nn

from .batching import PackedGroupedBatch
from .forecasting import packed_last_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMMTSF_ROOT = PROJECT_ROOT / "external" / "IMM-TSF"
if str(IMMTSF_ROOT) not in sys.path:
    sys.path.insert(0, str(IMMTSF_ROOT))

from fusions.MMF_GR_Add import MMF_GR_Add  # noqa: E402
from fusions.MMF_XAttn_Add import MMF_XAttn_Add  # noqa: E402
from fusions.TTF_RecAvg import TTF_RecAvg  # noqa: E402
from fusions.TTF_T2V_XAttn import TTF_T2V_XAttn  # noqa: E402


TimeMode = Literal["unit_consistent", "released_mixed_units"]


class IMMTSFRecAvgGRAddBaseline(nn.Module):
    """Official RecAvg + gated-residual fusion over numeric persistence.

    ``unit_consistent`` divides both text and query times by the complete window
    length. ``released_mixed_units`` reproduces the upstream data path, which
    divides query times but leaves text times in physical units.
    """

    def __init__(
        self,
        *,
        input_dims: tuple[int, ...],
        numeric_modalities: tuple[int, ...],
        text_modality: int,
        target_dim: int,
        total_window: float,
        time_mode: TimeMode = "unit_consistent",
        d_txt: int = 768,
        recency_sigma: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(numeric_modalities) != target_dim:
            raise ValueError("Numeric persistence count must match target dimension")
        if text_modality in numeric_modalities:
            raise ValueError("Text modality cannot be numeric")
        if input_dims[text_modality] != 768:
            raise ValueError("Official GPT-2 fusion expects 768-dimensional notes")
        if total_window <= 0:
            raise ValueError("total_window must be positive")
        if time_mode not in ("unit_consistent", "released_mixed_units"):
            raise ValueError(f"Unknown time mode: {time_mode}")
        self.input_dims = input_dims
        self.numeric_modalities = numeric_modalities
        self.text_modality = text_modality
        self.target_dim = target_dim
        self.total_window = float(total_window)
        self.time_mode = time_mode
        self.ttf = TTF_RecAvg(
            "GPT2",
            6,
            max_length=320,
            device="cpu",
            use_text_embeddings=True,
            recency_sigma=recency_sigma,
            dropout=dropout,
            d_txt=d_txt,
        )
        self.mmf = MMF_GR_Add(
            d_txt=d_txt,
            C=target_dim,
            hidden_dim=target_dim,
            dropout=dropout,
        )

    def _packed_text(
        self, packed: PackedGroupedBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = packed.values[self.text_modality]
        mask = packed.event_masks[self.text_modality]
        batch_size, group_count, per_group, width = values.shape
        notes = values.reshape(batch_size, group_count * per_group, width)
        note_mask = mask.reshape(batch_size, group_count * per_group)
        tau = (
            packed.arrival_times.unsqueeze(2)
            .expand(-1, -1, per_group)
            .reshape(batch_size, group_count * per_group)
        )
        return notes, tau, note_mask

    def _scale_times(
        self, tau: torch.Tensor, query_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_queries = query_times / self.total_window
        if self.time_mode == "released_mixed_units":
            return tau, normalized_queries
        return tau / self.total_window, normalized_queries

    def _recency_text(
        self,
        notes: torch.Tensor,
        note_mask: torch.Tensor,
        tau: torch.Tensor,
        query_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Official TTF_RecAvg forward with an explicit nonsemantic mask."""

        values = notes
        if self.ttf.input_proj is not None:
            values = self.ttf.input_proj(values)
        delta = (query_times[:, None] - tau[:, :, None]).clamp_min(0)
        sigma = self.ttf.log_recency_sigma.exp()
        weights = torch.exp(-((delta / sigma) ** 2))
        weights = weights * note_mask.to(weights.dtype)[:, :, None]
        weighted_sum = torch.einsum("bnt,bnd->btd", weights, values)
        denominator = weights.sum(dim=1).clamp_min(1e-6)
        text = weighted_sum / denominator.unsqueeze(-1)
        text = self.ttf.layer_norm(text)
        text = self.ttf.dropout(text)
        text = self.ttf.proj(text)
        presence = note_mask.any(dim=1, keepdim=True)
        return text, presence

    def forward_packed_queries(
        self, packed: PackedGroupedBatch, query_times: torch.Tensor
    ) -> torch.Tensor:
        query_times = query_times.to(packed.arrival_times)
        persistence = packed_last_values(
            packed, self.input_dims, self.numeric_modalities
        )
        notes, tau, note_mask = self._packed_text(packed)
        scaled_tau, scaled_queries = self._scale_times(tau, query_times)
        text, presence = self._recency_text(
            notes, note_mask, scaled_tau, scaled_queries
        )
        base = persistence.unsqueeze(1).expand(-1, query_times.shape[1], -1)
        return self.mmf(base, text, presence)

    def forward_packed(self, packed: PackedGroupedBatch) -> torch.Tensor:
        return self.forward_packed_queries(
            packed, packed.query_times.unsqueeze(1)
        )[:, 0]


class IMMTSFT2VXAttnBaseline(nn.Module):
    """Official T2V cross-attention + cross-attention-add fusion.

    The upstream modules infer padding from zero-valued embeddings.  That would
    erase event presence in a zero-content control, so this wrapper executes the
    same public layers with the packed event mask supplied explicitly.  Thus the
    semantic ablation preserves timestamps and note counts.
    """

    def __init__(
        self,
        *,
        input_dims: tuple[int, ...],
        numeric_modalities: tuple[int, ...],
        text_modality: int,
        target_dim: int,
        total_window: float,
        d_txt: int = 768,
        n_heads: int = 1,
        dropout: float = 0.1,
        kappa: float = 1.0,
    ) -> None:
        super().__init__()
        if len(numeric_modalities) != target_dim:
            raise ValueError("Numeric persistence count must match target dimension")
        if input_dims[text_modality] != 768:
            raise ValueError("Official GPT-2 fusion expects 768-dimensional notes")
        if total_window <= 0:
            raise ValueError("total_window must be positive")
        self.input_dims = input_dims
        self.numeric_modalities = numeric_modalities
        self.text_modality = text_modality
        self.target_dim = target_dim
        self.total_window = float(total_window)
        self.ttf = TTF_T2V_XAttn(
            "GPT2",
            6,
            max_length=320,
            device="cpu",
            use_text_embeddings=True,
            n_heads_fusion=n_heads,
            dropout=dropout,
            d_txt=d_txt,
        )
        self.mmf = MMF_XAttn_Add(
            d_txt=d_txt,
            C=target_dim,
            d_attn=d_txt,
            n_heads_fusion=n_heads,
            dropout=dropout,
            kappa=kappa,
        )

    def _packed_text(
        self, packed: PackedGroupedBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = packed.values[self.text_modality]
        mask = packed.event_masks[self.text_modality]
        batch_size, group_count, per_group, width = values.shape
        notes = values.reshape(batch_size, group_count * per_group, width)
        note_mask = mask.reshape(batch_size, group_count * per_group)
        tau = (
            packed.arrival_times.unsqueeze(2)
            .expand(-1, -1, per_group)
            .reshape(batch_size, group_count * per_group)
        )
        return notes, tau, note_mask

    def _time_text(
        self,
        notes: torch.Tensor,
        tau: torch.Tensor,
        note_mask: torch.Tensor,
        query_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = notes
        if self.ttf.input_proj is not None:
            values = self.ttf.input_proj(values)
        batch_size = values.shape[0]
        tau_feat = self.ttf.time2vec(tau.unsqueeze(-1))
        key_values = self.ttf.KV_proj(torch.cat([values, tau_feat], dim=-1))
        queries = self.ttf.Q_param.expand(
            batch_size, query_times.shape[1], self.ttf.d_txt
        )
        query_flat = queries.reshape(
            batch_size * query_times.shape[1], 1, self.ttf.d_txt
        )
        key_value_flat = (
            key_values.unsqueeze(1)
            .expand(-1, query_times.shape[1], -1, -1)
            .reshape(
                batch_size * query_times.shape[1],
                key_values.shape[1],
                self.ttf.d_txt,
            )
        )
        pad_flat = (
            (~note_mask)
            .unsqueeze(1)
            .expand(-1, query_times.shape[1], -1)
            .reshape(batch_size * query_times.shape[1], key_values.shape[1])
        )
        attended, _ = self.ttf.attn(
            query_flat,
            key_value_flat,
            key_value_flat,
            key_padding_mask=pad_flat,
        )
        attended = attended.reshape(
            batch_size, query_times.shape[1], self.ttf.d_txt
        )
        presence = note_mask.any(dim=1, keepdim=True)
        present_3d = presence.view(batch_size, 1, 1)
        attended = torch.where(present_3d, attended, torch.zeros_like(attended))
        residual = self.ttf.layer_norm(attended + queries)
        text = self.ttf.proj_out(self.ttf.dropout(residual))
        return text, presence

    def forward_packed_queries(
        self, packed: PackedGroupedBatch, query_times: torch.Tensor
    ) -> torch.Tensor:
        query_times = query_times.to(packed.arrival_times)
        persistence = packed_last_values(
            packed, self.input_dims, self.numeric_modalities
        )
        notes, tau, note_mask = self._packed_text(packed)
        scaled_tau = tau / self.total_window
        scaled_queries = query_times / self.total_window
        text, presence = self._time_text(
            notes, scaled_tau, note_mask, scaled_queries
        )
        base = persistence.unsqueeze(1).expand(-1, query_times.shape[1], -1)
        return self.mmf(base, text, presence)

    def forward_packed(self, packed: PackedGroupedBatch) -> torch.Tensor:
        return self.forward_packed_queries(
            packed, packed.query_times.unsqueeze(1)
        )[:, 0]
