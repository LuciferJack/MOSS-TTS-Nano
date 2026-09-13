import unittest

import torch
import torch.nn.functional as F

from finetuning.sft import compute_text_loss


SLOT = 9
STOP = 7


def logits_for(targets: torch.Tensor, *, continuation_logit: float = 2.0,
               stop_logit: float = 0.5) -> torch.Tensor:
    logits = torch.zeros((*targets.shape, 10))
    for batch in range(targets.shape[0]):
        for pos in range(targets.shape[1]):
            if targets[batch, pos] == SLOT:
                logits[batch, pos, SLOT] = continuation_logit
            elif targets[batch, pos] == STOP:
                logits[batch, pos, STOP] = stop_logit
    return logits


def compute(logits, targets, **kwargs):
    return compute_text_loss(
        logits, targets, audio_end_token_id=STOP,
        audio_assistant_slot_token_id=SLOT, **kwargs
    )


class EosLossTest(unittest.TestCase):
    def test_default_is_exact_legacy_cross_entropy_with_padding(self):
        targets = torch.tensor([[SLOT, SLOT, SLOT, STOP, -100]])
        logits = logits_for(targets)
        actual, _ = compute(logits, targets)
        expected = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1), ignore_index=-100)
        torch.testing.assert_close(actual, expected)

    def test_token_weight_applies_only_to_stop_target(self):
        targets = torch.tensor([[SLOT, SLOT, SLOT, STOP, -100]])
        logits = logits_for(targets)
        actual, breakdown = compute(logits, targets, eos_loss_weight=3.0)
        token_losses = F.cross_entropy(
            logits.reshape(-1, 10), targets.reshape(-1), ignore_index=-100, reduction="none"
        ).reshape_as(targets)
        expected = (token_losses[0, :3].sum() + 3.0 * token_losses[0, 3]) / 6.0
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(breakdown["text_continue_loss"], token_losses[0, :3].mean())
        torch.testing.assert_close(breakdown["text_stop_loss"], token_losses[0, 3])

    def test_sequence_balanced_is_length_independent_and_padding_safe(self):
        targets = torch.tensor([
            [SLOT, STOP, -100, -100, -100, -100],
            [SLOT, SLOT, SLOT, SLOT, SLOT, STOP],
        ])
        logits = logits_for(targets)
        loss, breakdown = compute(logits, targets, eos_loss_mode="sequence_balanced")
        continuation_row = torch.zeros((1, 10)); continuation_row[0, SLOT] = 2.0
        stop_row = torch.zeros((1, 10)); stop_row[0, STOP] = 0.5
        continuation_ce = F.cross_entropy(continuation_row, torch.tensor([SLOT]))
        stop_ce = F.cross_entropy(stop_row, torch.tensor([STOP]))
        torch.testing.assert_close(loss, (continuation_ce + stop_ce) / 2)
        self.assertEqual(breakdown["text_sequence_count"].item(), 2)
        self.assertEqual(breakdown["text_continue_count"].item(), 6)
        self.assertEqual(breakdown["text_stop_count"].item(), 2)

    def test_sequence_balanced_explicit_stop_coefficient(self):
        targets = torch.tensor([[SLOT, SLOT, STOP]])
        logits = logits_for(targets)
        loss, breakdown = compute(
            logits, targets, eos_loss_mode="sequence_balanced", eos_loss_weight=3.0
        )
        expected = (breakdown["text_continue_loss"] + 3 * breakdown["text_stop_loss"]) / 4
        torch.testing.assert_close(loss, expected)

    def test_missing_or_multiple_stop_fails_closed(self):
        for targets, count in (
            (torch.tensor([[SLOT, SLOT, -100]]), 0),
            (torch.tensor([[SLOT, STOP, STOP]]), 2),
        ):
            with self.subTest(stop_count=count):
                with self.assertRaisesRegex(ValueError, "exactly one audio_end"):
                    compute(logits_for(targets), targets)

    def test_unexpected_supervised_text_token_fails_closed(self):
        targets = torch.tensor([[SLOT, 4, STOP]])
        with self.assertRaisesRegex(ValueError, "other than slot/end"):
            compute(logits_for(targets), targets)

    def test_diagnostics_match_runtime_binary_boundary_not_vocab_argmax(self):
        targets = torch.tensor([[SLOT, STOP]])
        logits = logits_for(targets)
        logits[0, 0, 0] = 20.0
        _, breakdown = compute(logits, targets)
        self.assertEqual(breakdown["text_continue_accuracy"].item(), 1)
        self.assertEqual(breakdown["text_stop_accuracy"].item(), 1)
        self.assertGreater(breakdown["text_continue_margin"].item(), 0)
        self.assertGreater(breakdown["text_stop_margin"].item(), 0)


if __name__ == "__main__":
    unittest.main()
