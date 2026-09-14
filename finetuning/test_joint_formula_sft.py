from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


def write_report(path, weight):
    path.write_text(json.dumps({"acoustic_total_weight": weight, "verdict": "content_fail",
                                "candidate_weights_sha256": "b" * 64,
                                "content_eval_manifest_sha256": "c" * 64}), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    def test_v2_requires_hashed_prior_content_failure(self):
        train = rows()
        with TemporaryDirectory() as directory:
            report = Path(directory) / "prior.json"
            report.write_text(json.dumps({"acoustic_total_weight": 0.125,
                                          "verdict": "content_fail",
                                          "candidate_weights_sha256": "b" * 64,
                                          "content_eval_manifest_sha256": "c" * 64}), encoding="utf-8")
            options = kwargs(train)
            options["channel_weights"] = [1] + [0.5 / 16] * 16
            options["prior_report_path"] = str(report)
            options["prior_report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
            validate_joint_formula_pilot(train, [{"id": i} for i in range(8)], **options)
            for key, value in (("prior_report_sha256", "0" * 64),
                               ("prior_report_path", "missing.json")):
                broken = dict(options); broken[key] = value
                with self.assertRaises(ValueError):
                    validate_joint_formula_pilot(train, [{"id": i} for i in range(8)], **broken)
            arbitrary = dict(options); arbitrary["channel_weights"] = [1] + [0.25 / 16] * 16
            with self.assertRaisesRegex(ValueError, "0.125 or 0.5"):
                validate_joint_formula_pilot(train, [{"id": i} for i in range(8)], **arbitrary)

    def test_v3_requires_68_frame_boundaries_and_hashed_evidence(self):
        train = rows()
        for row in train:
            row["audio_codes"] = [[index % 17] * 16 for index in range(68)]
            row["audio_codes_sha256"] = digest(row["audio_codes"])
            row["acoustic_tail_start_frame"] = 34
        with TemporaryDirectory() as directory:
            root = Path(directory)
            v1, v2, audit, alignment = root / "v1.json", root / "v2.json", root / "audit.md", root / "alignment.json"
            v1_sha = write_report(v1, 0.125)
            v2_sha = write_report(v2, 0.5)
            audit.write_text("joint-v1-v2 trace audit\n", encoding="utf-8")
            alignment.write_text(json.dumps({
                "schema": "joint-teacher-tail-alignment-audit-v1",
                "rows": [{"id": row["id"], "audio_frames": 68,
                          "boundary_interval_frames": [34, 35]} for row in train],
            }), encoding="utf-8")
            alignment_sha = hashlib.sha256(alignment.read_bytes()).hexdigest()
            for row in train:
                row["acoustic_tail_alignment"] = {
                    "report_sha256": alignment_sha, "policy": "conservative_lower_bound",
                    "boundary_interval_frames": [34, 35], "selected_frame": 34,
                    "uncertainty_frames": 1,
                }
            options = kwargs(train)
            options.update(
                channel_weights=[1] + [0.5 / 16] * 16,
                prior_report_path=str(v1), prior_report_sha256=v1_sha,
                tail_weighting=True,
                v2_fail_report_path=str(v2), v2_fail_report_sha256=v2_sha,
                trace_audit_path=str(audit),
                trace_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
                alignment_report_path=str(alignment), alignment_report_sha256=alignment_sha,
            )
            validate_joint_formula_pilot(train, [{"id": i} for i in range(8)], **options)
            for value in (0, 68, True, None):
                broken_rows = deepcopy(train)
                if value is None:
                    broken_rows[0].pop("acoustic_tail_start_frame")
                else:
                    broken_rows[0]["acoustic_tail_start_frame"] = value
                with self.assertRaises(ValueError):
                    validate_joint_formula_pilot(
                        broken_rows, [{"id": i} for i in range(8)], **{**options,
                            "schedule": [row["id"] for row in broken_rows]}
                    )
            bad_hash = dict(options); bad_hash["trace_audit_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                validate_joint_formula_pilot(train, [{"id": i} for i in range(8)], **bad_hash)
            wide = json.loads(alignment.read_text())
            wide["rows"][0]["boundary_interval_frames"] = [34, 36]
            alignment.write_text(json.dumps(wide), encoding="utf-8")
            wide_sha = hashlib.sha256(alignment.read_bytes()).hexdigest()
            train[0]["acoustic_tail_alignment"].update(
                report_sha256=wide_sha, boundary_interval_frames=[34, 36], uncertainty_frames=2
            )
            wide_options = dict(options, alignment_report_sha256=wide_sha)
            with self.assertRaisesRegex(ValueError, "exceeds one frame"):
                validate_joint_formula_pilot(train, [{"id": i} for i in range(8)], **wide_options)

    def test_boundary_without_explicit_v3_is_rejected(self):
        train = rows(); train[0]["acoustic_tail_start_frame"] = 1
        with self.assertRaisesRegex(ValueError, "explicit v3"):
            validate_joint_formula_pilot(train, [{"id": i} for i in range(8)], **kwargs(train))


if __name__ == "__main__":
    unittest.main()
