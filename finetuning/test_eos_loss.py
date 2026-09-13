import unittest

import torch
import torch.nn.functional as F

from finetuning.sft import compute_text_loss


class EosLossTest(unittest.TestCase):
    def setUp(self):
        # Targets contain three continuation slots and one audio-end token.
        self.targets = torch.tensor([9, 9, 9, 7, -100])
        self.logits = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

    def test_default_is_exact_legacy_cross_entropy(self):
        actual, _ = compute_text_loss(
            self.logits, self.targets, audio_end_token_id=7, audio_assistant_slot_token_id=9
        )
        expected = F.cross_entropy(self.logits, self.targets, ignore_index=-100)
        torch.testing.assert_close(actual, expected)

    def test_weight_applies_only_to_stop_target(self):
        actual, breakdown = compute_text_loss(
            self.logits, self.targets, audio_end_token_id=7,
            audio_assistant_slot_token_id=9, eos_loss_weight=3.0
        )
        token_losses = F.cross_entropy(self.logits, self.targets, ignore_index=-100, reduction="none")
        expected = (token_losses[:3].sum() + 3.0 * token_losses[3]) / 6.0
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(breakdown["text_continue_loss"], token_losses[:3].mean())
        torch.testing.assert_close(breakdown["text_stop_loss"], token_losses[3])
        self.assertEqual(breakdown["text_continue_count"].item(), 3)
        self.assertEqual(breakdown["text_stop_count"].item(), 1)
        self.assertEqual(breakdown["text_continue_accuracy"].item(), 1)
        self.assertEqual(breakdown["text_stop_accuracy"].item(), 1)

    def test_missing_stop_is_reported_without_changing_continuation_mean(self):
        targets = torch.tensor([9, 9, -100, -100, -100])
        actual, breakdown = compute_text_loss(
            self.logits, targets, audio_end_token_id=7,
            audio_assistant_slot_token_id=9, eos_loss_weight=16.0
        )
        expected = F.cross_entropy(self.logits, targets, ignore_index=-100)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isnan(breakdown["text_stop_loss"]))
        self.assertEqual(breakdown["text_stop_count"].item(), 0)

    def test_accuracy_matches_runtime_binary_boundary_not_vocab_argmax(self):
        logits = self.logits.clone()
        logits[0, 0] = 20.0  # Unrelated vocabulary token wins full argmax.
        _, breakdown = compute_text_loss(
            logits, self.targets, audio_end_token_id=7, audio_assistant_slot_token_id=9
        )
        self.assertEqual(breakdown["text_continue_accuracy"].item(), 1)
        self.assertGreater(breakdown["text_continue_margin"].item(), 0)


if __name__ == "__main__":
    unittest.main()
