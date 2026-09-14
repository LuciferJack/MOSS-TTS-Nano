from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from finetuning.same_speaker_pilot import GLOBAL_LORA_TARGETS, HUMAN_LEDGER_SHA256, SOURCE_GATE_SHAS, SOURCE_MANIFEST_SHA256, canonical_sha, validate_same_speaker_pilot
from finetuning.same_speaker_pilot import HELDOUT_TEXT, TRAIN_TEXT


def row(identifier, role, category, asset, structure, reference_codes, reference_asset):
    codes = [[1] * 16, [2] * 16]
    result = {
        "id": identifier, "speaker": "Junhao", "category": category,
        "text": identifier, "structure_signature": structure,
        "audio_codes": codes, "audio_codes_sha256": canonical_sha(codes),
        "audio_asset_sha256": asset, "ref_audio_codes": reference_codes,
        "reference_codes_sha256": canonical_sha(reference_codes),
        "reference_provenance": {"speaker": "Junhao", "audio_asset_sha256": reference_asset},
        "gradient_eligible": role != "same_speaker_professional_heldout",
    }
    if "protector" in role:
        result["protector_role"] = role
    else:
        result["training_role"] = role
    return result


class SameSpeakerPilotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.root = Path(self.tmp.name)
        (self.root / "pytorch_model.bin").write_bytes(b"clean-base-test")
        self.base_sha = hashlib.sha256((self.root / "pytorch_model.bin").read_bytes()).hexdigest()
        self.ref_codes = [[9] * 16]
        self.ref_asset = "f" * 64
        self.train = [row("junhao_caoh", "same_speaker_professional_train", "professional", "a" * 64,
                          "parentheses", self.ref_codes, self.ref_asset)]
        self.heldout = [row("junhao_hydrate", "same_speaker_professional_heldout", "professional", "b" * 64,
                            "hydrate-dot", self.ref_codes, self.ref_asset)]
        self.train[0]["text"] = TRAIN_TEXT
        self.heldout[0]["text"] = HELDOUT_TEXT
        self.protect = [row(f"natural-{i}", "same_speaker_natural_protector", "natural",
                            f"{i + 1:x}" * 64, f"natural-{i}", self.ref_codes, self.ref_asset)
                        for i in range(5)]
        assets = [{"id": r["id"], "audio_sha256": r["audio_asset_sha256"],
                   "codec_codes_canonical_json_sha256": r["audio_codes_sha256"],
                   "source_gate_artifact_sha256": SOURCE_GATE_SHAS[r["id"]],
                   "human_gate_ledger_sha256": HUMAN_LEDGER_SHA256,
                   "reference": {"audio_sha256": self.ref_asset,
                                  "codec_codes_canonical_json_sha256": canonical_sha(self.ref_codes)}}
                  for r in self.train + self.heldout]
        natural = [{"id": r["id"], "audio_sha256": r["audio_asset_sha256"],
                    "codec_codes_sha256": r["audio_codes_sha256"]} for r in self.protect]
        self.gate = self.root / "manifest.json"
        self.gate.write_text(json.dumps({"schema": "min-same-voice-causal-pilot-preflight-v1",
                                         "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                                         "assets": assets, "natural_pcgrad_only": natural}), encoding="utf-8")
        self.gate_sha = hashlib.sha256(self.gate.read_bytes()).hexdigest()

    def tearDown(self): self.tmp.cleanup()

    def options(self):
        return dict(preflight_manifest_path=str(self.gate), preflight_manifest_sha256=self.gate_sha,
                    max_train_steps=4, eos_loss_mode="sequence_balanced",
                    channel_weights=[1] + [0.125 / 16] * 16,
                    protect_weights=[0] + [1 / 16] * 16, pcgrad=True, lora_rank=4,
                    lora_target_modules=GLOBAL_LORA_TARGETS, model_path=str(self.root))

    def validate(self, train=None, heldout=None, protect=None, **changes):
        opts = self.options(); opts.update(changes)
        with patch("finetuning.same_speaker_pilot.PREFLIGHT_SHA256", self.gate_sha), patch(
            "finetuning.same_speaker_pilot.BASE_MODEL_SHA256",
            self.base_sha,
        ):
            return validate_same_speaker_pilot(train or self.train, heldout or self.heldout,
                                               protect or self.protect, **opts)

    def test_exact_contract_and_early_stop_candidates(self):
        self.assertEqual(self.validate()["candidate_steps"], [1, 2, 4])
        self.assertEqual(self.validate(max_train_steps=2)["candidate_steps"], [1, 2])

    def test_heldout_never_gradient_or_protection(self):
        heldout = deepcopy(self.heldout); heldout[0]["gradient_eligible"] = True
        with self.assertRaisesRegex(ValueError, "gradient_eligible=false"):
            self.validate(heldout=heldout)
        with self.assertRaises(ValueError):
            self.validate(protect=self.protect[:4] + self.heldout)
        leaked = deepcopy(self.protect)
        leaked[0]["audio_asset_sha256"] = self.heldout[0]["audio_asset_sha256"]
        with self.assertRaisesRegex(ValueError, "heldout/protector leakage"):
            self.validate(protect=leaked)

    def test_reference_is_explicit_shared_and_independent(self):
        broken = deepcopy(self.heldout); broken[0]["ref_audio_codes"] = [[8] * 16]
        broken[0]["reference_codes_sha256"] = canonical_sha(broken[0]["ref_audio_codes"])
        with self.assertRaisesRegex(ValueError, "exact same explicit reference"):
            self.validate(heldout=broken)
        broken_train, broken_heldout, broken_protect = deepcopy(self.train), deepcopy(self.heldout), deepcopy(self.protect)
        for item in broken_train + broken_heldout + broken_protect:
            item["reference_provenance"]["audio_asset_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "independent"):
            self.validate(train=broken_train, heldout=broken_heldout, protect=broken_protect)

    def test_machine_artifact_and_ledger_hashes_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.validate(preflight_manifest_sha256="0" * 64)
        report = json.loads(self.gate.read_text()); report["assets"][0]["human_gate_ledger_sha256"] = "0" * 64
        self.gate.write_text(json.dumps(report), encoding="utf-8")
        self.gate_sha = hashlib.sha256(self.gate.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "machine artifacts/ledger"):
            self.validate()

    def test_failed_objectives_and_professional_protector_are_rejected(self):
        for changes in ({"max_train_steps": 3}, {"eos_loss_mode": "token_weight"},
                        {"tail_weighting": True}, {"channel_weights": [1] + [0.5 / 16] * 16},
                        {"pcgrad": False}, {"lora_target_modules": "local_transformer"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate(**changes)
        protect = deepcopy(self.protect); protect[0]["category"] = "professional"
        with self.assertRaisesRegex(ValueError, "Professional examples"):
            self.validate(protect=protect)
        bad_protect = [0] + [1 / 16] * 16; bad_protect[1:3] = [-1, 1 + 1 / 16]
        with self.assertRaisesRegex(ValueError, "acoustic-only"):
            self.validate(protect_weights=bad_protect)

    def test_pins_professional_ids_and_rejects_stacked_adapter(self):
        train = deepcopy(self.train); train[0]["id"] = "swapped"
        with self.assertRaisesRegex(ValueError, "pinned"):
            self.validate(train=train)
        train = deepcopy(self.train); train[0]["text"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "spoken targets"):
            self.validate(train=train)
        heldout = deepcopy(self.heldout); heldout[0].pop("structure_signature")
        with self.assertRaisesRegex(ValueError, "non-empty structure"):
            self.validate(heldout=heldout)
        (self.root / "adapter_config.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stacking"):
            self.validate(model_path=str(self.root))
        (self.root / "adapter_config.json").unlink()
        (self.root / "pytorch_model.bin").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "approved frozen clean base"):
            self.validate(model_path=str(self.root))


if __name__ == "__main__": unittest.main()
