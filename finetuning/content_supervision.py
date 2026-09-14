"""Independent content-head supervision (route A1, target-side content pivot).

Auxiliary frame-level character classifier attached to the global transformer's
per-frame hidden states. The head is a training-time representation shaper: its
gradient reaches the shared hidden states (and through them the codec heads)
via the jointly trained LoRA adapters, while the head itself is dropped at
inference. Labels come from a deterministic, auditable per-frame alignment
manifest; the manifest -- not this module -- is the pinned trust anchor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BLANK_TOKEN = "<blank>"


class ContentHead(nn.Module):
    """Small per-frame classifier: global hidden state -> character posterior."""

    def __init__(self, hidden_size: int, vocab: Sequence[str]):
        super().__init__()
        if len(vocab) == 0 or vocab[0] != BLANK_TOKEN:
            raise ValueError(f"content vocab must start with {BLANK_TOKEN!r}")
        if len(set(vocab)) != len(vocab):
            raise ValueError("content vocab contains duplicates")
        self.vocab = list(vocab)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, len(self.vocab))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shaped = self.norm(hidden_states)
        shaped = F.gelu(self.fc1(shaped))
        return self.fc2(shaped)


def _validate_alignment_record(record: dict, line_number: int) -> Tuple[str, int, Tuple[str, ...], Tuple[int, ...]]:
    sample_id = str(record.get("id", ""))
    if not sample_id:
        raise ValueError(f"alignment line {line_number}: missing 'id'")
    vocab = record.get("vocab")
    if not isinstance(vocab, list) or not vocab or vocab[0] != BLANK_TOKEN:
        raise ValueError(f"alignment line {line_number} ({sample_id}): vocab must be a list starting with {BLANK_TOKEN!r}")
    if len(set(vocab)) != len(vocab):
        raise ValueError(f"alignment line {line_number} ({sample_id}): duplicate vocab entries")
    frame_labels = record.get("frame_labels")
    frames = record.get("frames")
    if not isinstance(frame_labels, list) or not all(isinstance(label, int) for label in frame_labels):
        raise ValueError(f"alignment line {line_number} ({sample_id}): frame_labels must be a list of ints")
    if frames is None or int(frames) != len(frame_labels):
        raise ValueError(
            f"alignment line {line_number} ({sample_id}): frames={frames} does not match "
            f"len(frame_labels)={len(frame_labels)}"
        )
    if any(label < 0 or label >= len(vocab) for label in frame_labels):
        raise ValueError(f"alignment line {line_number} ({sample_id}): frame_labels out of vocab range")
    return sample_id, int(frames), tuple(str(token) for token in vocab), tuple(int(label) for label in frame_labels)


def load_content_alignment(path: str) -> Dict[str, dict]:
    """Load and validate the per-frame alignment manifest.

    Every record must share an identical vocab (single global classifier).
    Returns {sample_id: {"frames": int, "vocab": tuple, "frame_labels": tuple}}.
    """
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"alignment manifest {path} is empty")
    alignment: Dict[str, dict] = {}
    vocab_ref: Optional[Tuple[str, ...]] = None
    for line_number, line in enumerate(lines, start=1):
        record = json.loads(line)
        sample_id, frames, vocab, frame_labels = _validate_alignment_record(record, line_number)
        if sample_id in alignment:
            raise ValueError(f"alignment line {line_number}: duplicate id {sample_id!r}")
        if vocab_ref is None:
            vocab_ref = vocab
        elif vocab != vocab_ref:
            raise ValueError(f"alignment line {line_number} ({sample_id}): vocab differs from first record")
        alignment[sample_id] = {"frames": frames, "vocab": vocab, "frame_labels": frame_labels}
    return alignment


def alignment_vocab(alignment: Dict[str, dict]) -> List[str]:
    first = next(iter(alignment.values()))
    return list(first["vocab"])


def validate_alignment_covers_records(alignment: Dict[str, dict], records: Sequence[dict]) -> None:
    """Require every gradient-eligible train record to have a frame-exact alignment."""
    missing = []
    mismatched = []
    for record in records:
        if record.get("eligible_for_training", True) is False:
            continue
        sample_id = str(record.get("id", ""))
        entry = alignment.get(sample_id)
        if entry is None:
            missing.append(sample_id)
            continue
        audio_codes = record.get("audio_codes") or []
        if int(entry["frames"]) != len(audio_codes):
            mismatched.append(f"{sample_id}: alignment_frames={entry['frames']} audio_codes={len(audio_codes)}")
    if missing:
        raise ValueError(f"alignment manifest missing gradient-eligible train ids: {sorted(missing)}")
    if mismatched:
        raise ValueError(f"alignment frame count mismatch: {mismatched}")


def build_content_targets(
    sample_ids: Sequence[str],
    acoustic_frame_indices: torch.Tensor,
    alignment: Dict[str, dict],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Map batch response frames to per-frame character-class targets.

    Positions outside the response region (and the end row) stay -100.
    acoustic_frame_indices: [bsz, slen] with frame number or -1, as produced by
    the dataset collate_fn.
    """
    if acoustic_frame_indices.dim() != 2:
        raise ValueError("acoustic_frame_indices must be [bsz, slen]")
    batch_size, seq_len = acoustic_frame_indices.shape
    if len(sample_ids) != batch_size:
        raise ValueError("sample_ids length does not match batch size")
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    for batch_index, sample_id in enumerate(sample_ids):
        entry = alignment.get(str(sample_id))
        if entry is None:
            raise ValueError(f"teacher batch sample {sample_id!r} has no content alignment")
        frame_labels = entry["frame_labels"]
        frame_index_row = acoustic_frame_indices[batch_index]
        valid_positions = frame_index_row.ge(0)
        if not bool(valid_positions.any()):
            raise ValueError(f"teacher batch sample {sample_id!r} has no response frames")
        frame_numbers = frame_index_row[valid_positions]
        if int(frame_numbers.max().item()) >= len(frame_labels):
            raise ValueError(
                f"teacher batch sample {sample_id!r}: frame index exceeds alignment length "
                f"({int(frame_numbers.max().item())} >= {len(frame_labels)})"
            )
        targets[batch_index, valid_positions] = torch.as_tensor(
            [frame_labels[int(frame_number)] for frame_number in frame_numbers.tolist()],
            dtype=torch.long,
            device=device,
        )
    return targets


def compute_content_loss(
    content_head: ContentHead,
    global_hidden_states: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Masked per-frame CE over the response region; returns (loss, metrics)."""
    if global_hidden_states.shape[:2] != targets.shape:
        raise ValueError(
            f"hidden {tuple(global_hidden_states.shape[:2])} vs targets {tuple(targets.shape)} mismatch"
        )
    flat_hidden = global_hidden_states.reshape(-1, global_hidden_states.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets.ne(-100)
    if not bool(valid.any()):
        raise ValueError("content loss received no labelled response frames")
    logits = content_head(flat_hidden[valid]).float()
    labels = flat_targets[valid]
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        accuracy = logits.argmax(dim=-1).eq(labels).float().mean()
    return loss, {
        "content_loss": float(loss.detach().float()),
        "content_frames": int(valid.sum().item()),
        "content_accuracy": float(accuracy),
    }
