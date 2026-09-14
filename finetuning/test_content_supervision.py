"""Unit tests for finetuning.content_supervision (route A1 content head)."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from finetuning.content_supervision import (
    BLANK_TOKEN,
    ContentHead,
    alignment_vocab,
    build_content_targets,
    compute_content_loss,
    load_content_alignment,
    validate_alignment_covers_records,
)


def _alignment_record(sample_id="junhao_caoh", vocab=None, frame_labels=None, frames=None):
    return {
        "id": sample_id,
        "vocab": vocab if vocab is not None else [BLANK_TOKEN, "二", "钙"],
        "frame_labels": frame_labels if frame_labels is not None else [1, 2, 0, 1],
        "frames": frames if frames is not None else len(frame_labels or [1, 2, 0, 1]),
    }


class TestContentHead(unittest.TestCase):
    def test_forward_shape_and_vocab(self):
        head = ContentHead(hidden_size=16, vocab=[BLANK_TOKEN, "a", "b"])
        logits = head(torch.randn(5, 16))
        self.assertEqual(tuple(logits.shape), (5, 3))

    def test_vocab_must_start_with_blank(self):
        with self.assertRaises(ValueError):
            ContentHead(hidden_size=16, vocab=["a", "b"])

    def test_vocab_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            ContentHead(hidden_size=16, vocab=[BLANK_TOKEN, "a", "a"])


class TestLoadAlignment(unittest.TestCase):
    def _write(self, records):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "alignment.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        self.addCleanup(tmp.cleanup)
        return str(path)

    def test_load_ok_and_vocab(self):
        path = self._write([_alignment_record()])
        alignment = load_content_alignment(path)
        self.assertEqual(list(alignment), ["junhao_caoh"])
        self.assertEqual(alignment_vocab(alignment), [BLANK_TOKEN, "二", "钙"])
        self.assertEqual(alignment["junhao_caoh"]["frames"], 4)

    def test_rejects_vocab_without_blank(self):
        path = self._write([_alignment_record(vocab=["a", "b"])])
        with self.assertRaises(ValueError):
            load_content_alignment(path)

    def test_rejects_frame_count_mismatch(self):
        path = self._write([_alignment_record(frames=99)])
        with self.assertRaises(ValueError):
            load_content_alignment(path)

    def test_rejects_out_of_range_labels(self):
        path = self._write([_alignment_record(frame_labels=[1, 2, 5])])
        with self.assertRaises(ValueError):
            load_content_alignment(path)

    def test_rejects_duplicate_ids(self):
        path = self._write([_alignment_record(), _alignment_record()])
        with self.assertRaises(ValueError):
            load_content_alignment(path)

    def test_rejects_vocab_drift_between_records(self):
        other = _alignment_record(sample_id="other", vocab=[BLANK_TOKEN, "x"])
        path = self._write([_alignment_record(), other])
        with self.assertRaises(ValueError):
            load_content_alignment(path)

    def test_rejects_empty_file(self):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "alignment.jsonl"
        path.write_text("\n", encoding="utf-8")
        self.addCleanup(tmp.cleanup)
        with self.assertRaises(ValueError):
            load_content_alignment(str(path))


class TestValidateCoverage(unittest.TestCase):
    def test_covers_records(self):
        records = [{"id": "junhao_caoh", "audio_codes": [[0] * 16] * 4}]
        alignment = {"junhao_caoh": {"frames": 4, "vocab": (BLANK_TOKEN,), "frame_labels": (0, 0, 0, 0)}}
        validate_alignment_covers_records(alignment, records)

    def test_missing_train_id_fails(self):
        records = [{"id": "junhao_caoh", "audio_codes": [[0] * 16] * 4}]
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_alignment_covers_records({}, records)

    def test_frame_mismatch_fails(self):
        records = [{"id": "junhao_caoh", "audio_codes": [[0] * 16] * 5}]
        alignment = {"junhao_caoh": {"frames": 4, "vocab": (BLANK_TOKEN,), "frame_labels": (0, 0, 0, 0)}}
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_alignment_covers_records(alignment, records)

    def test_ineligible_records_skipped(self):
        records = [{"id": "junhao_hydrate", "eligible_for_training": False, "audio_codes": [[0] * 16] * 9}]
        validate_alignment_covers_records({}, records)


class TestBuildTargets(unittest.TestCase):
    def test_maps_response_frames_only(self):
        # slen 6; item 0: response frames at positions 2,3 (frame numbers 0,1);
        # item 1: response frames at position 1 (frame number 0).
        indices = torch.tensor([
            [-1, -1, 0, 1, -1, -1],
            [-1, 0, -1, -1, -1, -1],
        ])
        alignment = {
            "s0": {"frames": 2, "vocab": (BLANK_TOKEN, "a"), "frame_labels": (1, 0)},
            "s1": {"frames": 1, "vocab": (BLANK_TOKEN, "a"), "frame_labels": (1,)},
        }
        targets = build_content_targets(["s0", "s1"], indices, alignment, device=torch.device("cpu"))
        self.assertEqual(targets.tolist(), [
            [-100, -100, 1, 0, -100, -100],
            [-100, 1, -100, -100, -100, -100],
        ])

    def test_missing_sample_raises(self):
        indices = torch.tensor([[0]])
        with self.assertRaisesRegex(ValueError, "no content alignment"):
            build_content_targets(["ghost"], indices, {}, device=torch.device("cpu"))


class TestComputeContentLoss(unittest.TestCase):
    def test_masked_ce_matches_manual_value(self):
        torch.manual_seed(0)
        head = ContentHead(hidden_size=8, vocab=[BLANK_TOKEN, "a", "b"])
        hidden = torch.randn(1, 3, 8)
        # targets: pos0 -> class 1, pos1 ignored, pos2 -> class 2
        targets = torch.tensor([[1, -100, 2]])
        loss, metrics = compute_content_loss(head, hidden, targets)
        logits = head(hidden[0])
        manual = 0.5 * (
            torch.nn.functional.cross_entropy(logits[0:1], torch.tensor([1]))
            + torch.nn.functional.cross_entropy(logits[2:3], torch.tensor([2]))
        )
        self.assertTrue(torch.allclose(loss, manual, atol=1e-6))
        self.assertEqual(metrics["content_frames"], 2)
        self.assertTrue(0.0 <= metrics["content_accuracy"] <= 1.0)

    def test_all_ignored_raises(self):
        head = ContentHead(hidden_size=8, vocab=[BLANK_TOKEN, "a"])
        with self.assertRaisesRegex(ValueError, "no labelled"):
            compute_content_loss(head, torch.randn(1, 2, 8), torch.full((1, 2), -100))

    def test_shape_mismatch_raises(self):
        head = ContentHead(hidden_size=8, vocab=[BLANK_TOKEN, "a"])
        with self.assertRaises(ValueError):
            compute_content_loss(head, torch.randn(1, 2, 8), torch.tensor([[1, -100, 0]]))


if __name__ == "__main__":
    unittest.main()
