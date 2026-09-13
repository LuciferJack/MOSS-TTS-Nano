from __future__ import annotations

import unittest

from finetuning.duration_stop_controller import (
    ConservativeStopController, StopTrace, leave_one_formula_out,
)


def trace(case_id, formula, boundary):
    length = 120
    return StopTrace(case_id, f"{formula} 是化学式。", f"字母展开 {formula} 中文名称。", boundary,
                     tuple([-1.0] * length), tuple([-3.0] * length), length)


class DurationStopControllerTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            trace("water", "H₂O", 40), trace("lime", "Ca(OH)₂", 52),
            trace("glucose", "C₆H₁₂O₆", 65), trace("cobalt", "CoCl₂·6H₂O", 78),
            trace("copper", "CuSO₄·5H₂O", 80),
        ]

    def test_native_eos_always_wins(self):
        controller = ConservativeStopController(self.rows)
        window = controller.window("CuSO₄·5H₂O 是化学式。", "展开读法", max_frames=120)
        self.assertEqual(controller.decide(frame_index=1, slot_logit=-2, end_logit=-1, window=window),
                         "native_eos")

    def test_non_formula_and_low_confidence_never_override(self):
        controller = ConservativeStopController(self.rows)
        ordinary = controller.window("这是普通句子。", "这是普通句子。", max_frames=120)
        self.assertEqual(controller.decide(frame_index=119, slot_logit=1, end_logit=-1, window=ordinary),
                         "continue")
        for text in ("版本 2 已发布。", "请等待 10 秒。", "SDK2 已发布。", "SDK(2)", "GPT(4)"):
            self.assertFalse(controller.window(text, text, max_frames=120).formula_scoped)
        remote = controller.window("X₉₉₉₉₉₉₉₉₉₉ 是未知式。", "非常不同的长文本" * 20, max_frames=120)
        self.assertLess(remote.confidence, controller.minimum_confidence)
        self.assertEqual(controller.decide(frame_index=119, slot_logit=1, end_logit=-1, window=remote),
                         "continue")

    def test_controller_waits_for_safe_edge(self):
        controller = ConservativeStopController(self.rows)
        window = controller.window("CuSO₄·5H₂O 是化学式。", "字母展开 CuSO₄·5H₂O 中文名称。", max_frames=120)
        self.assertEqual(controller.decide(frame_index=window.safe_stop_frame - 1,
                                           slot_logit=1, end_logit=-1, window=window), "continue")
        self.assertEqual(controller.decide(frame_index=window.safe_stop_frame,
                                           slot_logit=1, end_logit=-1, window=window), "controller_eos")

    def test_leave_one_formula_out_reports_early_cut_explicitly(self):
        folds = leave_one_formula_out(self.rows)
        self.assertEqual({row["case_id"] for row in folds}, {row.case_id for row in self.rows})
        self.assertTrue(all(not row["early_cut"] for row in folds))
        self.assertTrue(all(row["before_cap"] for row in folds))

    def test_malformed_traces_fail_closed(self):
        broken = StopTrace("x", "H₂O", "H 二 O", 10, (1.0,), (1.0,), 120)
        with self.assertRaisesRegex(ValueError, "ends before"):
            ConservativeStopController([broken, trace("a", "CO₂", 20), trace("b", "NO₂", 22)])


if __name__ == "__main__":
    unittest.main()
