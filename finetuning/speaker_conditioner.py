"""Independent speaker conditioner (route B1): CAMPPlus-FiLM voice conditioning.

A small trainable adapter that injects a fixed speaker encoder's embedding
into the global transformer via per-block FiLM (scale/shift at block input).
The frozen CAMPPlus ONNX runs only at data-prep time; training and inference
consume the precomputed, hash-pinned embeddings manifest. The conditioner is
attached to the frozen base model with forward-pre-hooks, so the trust-remote
modeling code stays untouched.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

DEFAULT_EMBEDDING_DIM = 512


class SpeakerConditioner(nn.Module):
    """shared MLP (emb->hidden) + per-layer zero-init low-rank FiLM factors.

    Zero initialization makes scale=0/shift=0 at start, so the conditioned
    model is exactly the base model until training moves the factors.
    """

    def __init__(self, *, embedding_dim: int, hidden_size: int, n_layers: int, film_rank: int):
        super().__init__()
        if embedding_dim <= 0 or hidden_size <= 0 or n_layers <= 0 or film_rank <= 0:
            raise ValueError("speaker conditioner dims must be positive")
        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.film_rank = int(film_rank)
        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.GELU(),
        )
        self.scale_left = nn.Parameter(torch.zeros(self.n_layers, self.hidden_size, self.film_rank))
        self.scale_right = nn.Parameter(torch.zeros(self.n_layers, self.film_rank, self.hidden_size))
        self.shift_left = nn.Parameter(torch.zeros(self.n_layers, self.hidden_size, self.film_rank))
        self.shift_right = nn.Parameter(torch.zeros(self.n_layers, self.film_rank, self.hidden_size))
        self._batch_conditioning: Optional[torch.Tensor] = None

    def begin_batch(self, embeddings: torch.Tensor) -> None:
        """Cache the shared conditioning vector for the current forward batch.

        embeddings: [bsz, embedding_dim] raw (already L2-normalized) speaker vectors.
        """
        if embeddings.dim() != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"speaker embeddings must be [bsz, {self.embedding_dim}], got {tuple(embeddings.shape)}"
            )
        self._batch_conditioning = self.encoder(embeddings.to(dtype=self.encoder[0].weight.dtype))

    def end_batch(self) -> None:
        self._batch_conditioning = None

    def film(self, layer_index: int):
        """Return (scale, shift) of shape [bsz, 1, hidden] for one block."""
        cond = self._batch_conditioning
        if cond is None:
            raise RuntimeError("begin_batch must be called before film()")
        scale = (cond @ self.scale_left[layer_index]) @ self.scale_right[layer_index]
        shift = (cond @ self.shift_left[layer_index]) @ self.shift_right[layer_index]
        return scale.unsqueeze(1), shift.unsqueeze(1)


def attach_speaker_conditioner(model, conditioner: SpeakerConditioner) -> List:
    """Register FiLM forward-pre-hooks on every global transformer block.

    model may be a peft wrapper; blocks are resolved through the base model's
    `transformer.h`. Returns the hook handles (keep them to detach later).
    """
    base = model
    while hasattr(base, "base_model") and hasattr(base.base_model, "transformer") is False:
        base = base.base_model
    if not hasattr(base, "transformer") or not hasattr(base.transformer, "h"):
        base = getattr(base, "base_model", base)
    blocks = base.transformer.h
    if len(blocks) != conditioner.n_layers:
        raise ValueError(f"conditioner has {conditioner.n_layers} layers but model has {len(blocks)} blocks")
    handles = []

    def make_hook(layer_index: int):
        def hook(module, inputs):
            if conditioner._batch_conditioning is None:
                return None
            hidden = inputs[0]
            scale, shift = conditioner.film(layer_index)
            modified = hidden * (1.0 + scale.to(dtype=hidden.dtype)) + shift.to(dtype=hidden.dtype)
            return (modified,) + tuple(inputs[1:])
        return hook

    for index, block in enumerate(blocks):
        handles.append(block.register_forward_pre_hook(make_hook(index)))
    return handles


def load_speaker_embeddings(path: str) -> Dict[str, torch.Tensor]:
    """Load the pinned speaker-embeddings manifest.

    JSON object: {"references": {"<reference_id>": {"embedding": [floats], ...}}}
    Every embedding must be a finite 1-D float list; L2 norm is enforced == 1
    within tolerance because the manifest is produced by the L2-normalized
    CAMPPlus pipeline.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    references = raw.get("references")
    if not isinstance(references, dict) or not references:
        raise ValueError("speaker embeddings manifest must contain a non-empty 'references' object")
    embeddings: Dict[str, torch.Tensor] = {}
    dims: set[int] = set()
    for reference_id, entry in references.items():
        if not isinstance(entry, dict):
            raise ValueError(f"speaker embedding entry {reference_id!r} must be an object")
        vector = entry.get("embedding")
        if (not isinstance(vector, list) or not vector or
                any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in vector)):
            raise ValueError(f"speaker embedding {reference_id!r} must be a list of floats")
        tensor = torch.tensor([float(value) for value in vector], dtype=torch.float32)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"speaker embedding {reference_id!r} contains non-finite values")
        norm = float(tensor.norm())
        if not math.isclose(norm, 1.0, rel_tol=0, abs_tol=1e-3):
            raise ValueError(f"speaker embedding {reference_id!r} must be L2-normalized, got norm {norm:.6f}")
        dims.add(int(tensor.numel()))
        embeddings[str(reference_id)] = tensor
    if len(dims) != 1:
        raise ValueError(f"speaker embeddings must share one dimension, got {sorted(dims)}")
    return embeddings


def embedding_dim(embeddings: Dict[str, torch.Tensor]) -> int:
    return int(next(iter(embeddings.values())).numel())


def resolve_batch_embedding(
    sample_ids: Sequence[str],
    *,
    reference_id_by_sample: Dict[str, str],
    embeddings: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Stack per-row speaker embeddings for one batch."""
    rows = []
    for sample_id in sample_ids:
        reference_id = reference_id_by_sample.get(str(sample_id))
        if reference_id is None:
            raise ValueError(f"sample {sample_id!r} has no reference provenance for speaker conditioning")
        vector = embeddings.get(reference_id)
        if vector is None:
            raise ValueError(f"reference {reference_id!r} is missing from the speaker embeddings manifest")
        rows.append(vector)
    return torch.stack(rows, dim=0)
