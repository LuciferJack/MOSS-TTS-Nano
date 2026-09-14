from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import unittest

from finetuning.joint_formula_sft import validate_joint_formula_pilot


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def rows():
    result = []
    for index in range(5):
        codes = [[index + 1] * 16, [index + 2] * 16]
        result.append({"id": f"hydrate-{index}", "training_role": "authorized_formula_joint_sft",
                       "domain": "chemistry", "route": "formula_adapter",
                       "codec_loss_eligible": True, "canonical_text": f"H₂O-{index}",
                       "spoken_text": f"H 二 O {index}", "text": f"H 二 O {index}",
                       "audio_codes": codes, "audio_codes_sha256": digest(codes),
                       "teacher_provenance": {"authorization": "training-and-evaluation-authorized",
                                              "audio_asset_sha256": "a" * 64}})
    return result


def kwargs(train):
    return dict(enabled=True, pcgrad=True, protection_scope="formula_scoped", behavior_rows=[],
                eos_loss_mode="sequence_balanced", channel_weights=[1] + [0.125 / 16] * 16,
                acoustic_weights=[0] + [1 / 16] * 16,
                schedule=[row["id"] for row in train], max_train_steps=5, lora_rank=8)


class JointFormulaSFTTest(unittest.TestCase):
    def test_exact_contract_passes(self):
        train = rows(); validate_joint_formula_pilot(train, [{"id": i} for i in range(8)], **kwargs(train))

    def test_each_safety_boundary_fails_closed(self):
        train = rows()
        mutations = [
            lambda options: options.update(enabled=False), lambda options: options.update(pcgrad=False),
            lambda options: options.update(protection_scope="dual"),
            lambda options: options.update(behavior_rows=[{"id": "ordinary"}]),
            lambda options: options.update(eos_loss_mode="token_weight"),
            lambda options: options.update(channel_weights=[1] + [0] * 16),
            lambda options: options.update(channel_weights=[1] + [0.125 / 15] * 15 + [0]),
            lambda options: options.update(schedule=[]), lambda options: options.update(max_train_steps=10),
            lambda options: options.update(lora_rank=0),
            lambda options: options.update(lora_target_modules="local_transformer"),
            lambda options: options.update(lora_modules_to_save="audio_lm_heads"),
        ]
        for mutate in mutations:
            options = kwargs(train); mutate(options)
            with self.assertRaises(ValueError):
                validate_joint_formula_pilot(train, [{"id": i} for i in range(8)], **options)

    def test_provenance_and_codes_are_immutable(self):
        for mutate in (
            lambda row: row.update(text="different"),
            lambda row: row.update(audio_codes_sha256="0" * 64),
            lambda row: row["teacher_provenance"].update(authorization="unknown"),
            lambda row: row.update(route="base"),
        ):
            train = rows(); mutate(train[0])
            with self.assertRaises(ValueError):
                validate_joint_formula_pilot(train, [{"id": i} for i in range(8)], **kwargs(train))


if __name__ == "__main__":
    unittest.main()
