from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from finetuning.dataset import MossTTSNanoSFTDataset
from finetuning.sft import validate_args
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
        "reference_provenance": {"id": "junhao_real_a03", "speaker": "Junhao", "audio_asset_sha256": reference_asset},
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
        contract = self.validate()
        self.assertEqual(contract["candidate_steps"], [1, 2, 4])
        self.assertEqual(contract["train_id"], "junhao_caoh")
        self.assertEqual(contract["heldout_id"], "junhao_hydrate")
        self.assertNotIn(contract["heldout_id"], contract["protector_ids"])
        self.assertEqual(contract["reference_id"], "junhao_real_a03")
        self.assertEqual(self.validate(max_train_steps=2)["candidate_steps"], [1, 2])

    def test_train_batch_really_packs_exact_reference_codes(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(char) for char in text]
        config = SimpleNamespace(
            n_vq=16, im_start_token_id=1101, im_end_token_id=1102,
            audio_start_token_id=1103, audio_end_token_id=1104,
            audio_user_slot_token_id=1105, audio_assistant_slot_token_id=1106,
            audio_pad_token_id=0, pad_token_id=0,
        )
        dataset = MossTTSNanoSFTDataset(self.train, tokenizer=Tokenizer(), model_config=config, max_length=512)
        item = dataset[0]
        self.assertEqual(int(item["reference_frames"]), len(self.ref_codes))
        rows = item["full_input_ids"]
        reference_rows = rows[rows[:, 0] == config.audio_user_slot_token_id]
        self.assertTrue(torch.equal(reference_rows[:, 1:], torch.tensor(self.ref_codes)))

    def test_runtime_requires_individual_protectors_and_candidate_checkpoints(self):
        base = dict(max_length=256, per_device_batch_size=1, gradient_accumulation_steps=1,
                    learning_rate=1e-5, weight_decay=0, warmup_steps=0, warmup_ratio=0,
                    num_epochs=4, max_train_steps=4, max_grad_norm=1, logging_steps=1,
                    save_every_epochs=1, num_workers=0, lora_rank=4, lora_alpha=8,
                    eos_loss_weight=1, lora_dropout=0, pcgrad=True, protect_jsonl="protect",
                    behavior_protect_jsonl="", train_schedule_json="", same_speaker_pilot=True,
                    same_speaker_heldout_jsonl="heldout", same_speaker_preflight_manifest="manifest",
                    same_speaker_preflight_manifest_sha256="a" * 64, joint_formula_pilot=False)
        validate_args(SimpleNamespace(**base))
        for changes in ({"per_device_batch_size": 2}, {"save_every_epochs": 2}, {"num_epochs": 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_args(SimpleNamespace(**{**base, **changes}))

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
