from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from finetuning.sft import compute_tail_weighted_channel_loss


class AcousticTailLossTest(unittest.TestCase):
    def test_68_frames_16_heads_and_constant_loss_invariant(self):
        batch, frames, vocab = 1, 68, 7
        logits = torch.zeros(batch * frames, vocab)
        targets = torch.zeros(batch * frames, dtype=torch.long)
        indices = torch.arange(frames).reshape(batch, frames)
        boundaries = torch.tensor([34])
        legacy = F.cross_entropy(logits, targets)
        for _ in range(16):
            weighted = compute_tail_weighted_channel_loss(
                logits, targets, batch_size=batch, seq_len=frames,
                acoustic_frame_indices=indices, acoustic_tail_start_frames=boundaries,
            )
            self.assertTrue(torch.allclose(weighted, legacy, rtol=1e-7, atol=1e-7))
            self.assertAlmostEqual(weighted.item(), math.log(vocab), places=6)

    def test_tail_is_doubled_and_padding_is_excluded(self):
        logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [9.0, -9.0]])
        targets = torch.tensor([0, 0, 0, 0, -100])
        indices = torch.tensor([[0, 1, 2, 3, -1]])
        actual = compute_tail_weighted_channel_loss(
            logits, targets, batch_size=1, seq_len=5,
            acoustic_frame_indices=indices, acoustic_tail_start_frames=torch.tensor([2]),
        )
        losses = F.cross_entropy(logits[:4], targets[:4], reduction="none")
        expected = (losses[:2].sum() + 2 * losses[2:].sum()) / 6
        self.assertTrue(torch.allclose(actual, expected))

    def test_mixed_legacy_row_uses_unweighted_ce(self):
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]])
        targets = torch.zeros(4, dtype=torch.long)
        indices = torch.tensor([[0, 1], [0, 1]])
        actual = compute_tail_weighted_channel_loss(
            logits, targets, batch_size=2, seq_len=2,
            acoustic_frame_indices=indices,
            acoustic_tail_start_frames=torch.tensor([1, -1]),
            acoustic_tail_weighted_mask=torch.tensor([True, False]),
        )
        losses = F.cross_entropy(logits, targets, reduction="none").reshape(2, 2)
        expected = torch.stack(((losses[0, 0] + 2 * losses[0, 1]) / 3, losses[1].mean())).mean()
        self.assertTrue(torch.allclose(actual, expected))

    def test_bad_shapes_and_empty_sample_fail_closed(self):
        logits = torch.zeros(2, 2)
        targets = torch.full((2,), -100)
        with self.assertRaises(ValueError):
            compute_tail_weighted_channel_loss(
                logits, targets, batch_size=1, seq_len=2,
                acoustic_frame_indices=torch.tensor([[0, 1]]),
                acoustic_tail_start_frames=torch.tensor([1]),
            )
        with self.assertRaises(ValueError):
            compute_tail_weighted_channel_loss(
                logits, torch.zeros(2, dtype=torch.long), batch_size=1, seq_len=2,
                acoustic_frame_indices=torch.tensor([0, 1]),
                acoustic_tail_start_frames=torch.tensor([1]),
            )


if __name__ == "__main__":
    unittest.main()
