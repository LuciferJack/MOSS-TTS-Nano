from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
import unittest

from finetuning.dataset import MossTTSNanoSFTDataset
from finetuning.eos_calibration import validate_calibration_record, validate_calibration_records
from finetuning.sft import validate_calibration_objective


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def sha_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_json(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calibration_row():
    prompt = "六水合氯化钴。"
    generation = {"do_sample": False, "max_new_tokens": 120, "temperature": 1.0}
    prefix_codes = [[11, 12], [21, 22], [31, 32]]
    reference_codes = [[61, 62], [71, 72]]
    return {
        "id": "hydrate-on-policy-1", "text": "六水合氯化钴。",
        "teacher_role": "self_generated_prefix_eos_calibration",
        "audio_code_source": "self_generated_junhao_prefix", "prefix_voice": "Junhao",
        "codec_loss_eligible": False,
        "self_generated_prefix_codes": prefix_codes,
        "prefix_codes_sha256": sha_json(prefix_codes),
        "ref_audio_codes": reference_codes,
        "reference_codes_sha256": sha_json(reference_codes),
        "target_boundary_frame": 3, "prefix_generator_revision": "moss-nano@c55e552",
        "prefix_generator_model_sha256": "a" * 64,
        "generation_prompt": prompt, "prompt_sha256": sha_text(prompt),
        "generation_config": generation, "generation_config_sha256": sha_json(generation),
        "target_boundary_provenance": {
            "teacher_asset_sha256": "b" * 64, "provider": "authorized-external-teacher",
            "authorization": "training-and-evaluation-authorized",
        },
    }


class EOSCalibrationTest(unittest.TestCase):
    @staticmethod
    def config():
        return SimpleNamespace(
            n_vq=2, im_start_token_id=101, im_end_token_id=102,
            audio_start_token_id=103, audio_end_token_id=104,
            audio_user_slot_token_id=105, audio_assistant_slot_token_id=106,
            audio_pad_token_id=0, pad_token_id=0,
        )

    def dataset(self, rows):
        return MossTTSNanoSFTDataset(
            rows, tokenizer=CharacterTokenizer(), model_config=self.config(), max_length=256
        )

    def test_generated_codes_are_context_but_never_acoustic_labels(self):
        dataset = self.dataset([calibration_row()])
        batch = dataset.collate_fn([dataset[0]])
        self.assertTrue(any(value in (11, 21, 31) for value in batch["input_ids"][0, :, 1].tolist()))
        self.assertTrue((batch["labels"][0, :, 1:] == -100).all().item())
        text_labels = batch["labels"][0, :, 0].tolist()
        self.assertEqual(text_labels.count(self.config().audio_assistant_slot_token_id), 3)
        self.assertEqual(text_labels.count(self.config().audio_end_token_id), 1)

    def test_ordinary_replay_keeps_acoustic_labels_in_mixed_batch(self):
        replay = {"id": "acronym-replay", "text": "GPU", "audio_codes": [[41, 42], [51, 52]]}
        dataset = self.dataset([calibration_row(), replay])
        batch = dataset.collate_fn([dataset[0], dataset[1]])
        self.assertTrue((batch["labels"][0, :, 1:] == -100).all().item())
        self.assertTrue((batch["labels"][1, :, 1:] != -100).any().item())

    def test_objective_is_strictly_text_only_and_sequence_balanced(self):
        rows = [calibration_row()]
        validate_calibration_objective(rows, eos_loss_mode="sequence_balanced", channelwise_loss_weight=[1, 0, 0])
        for mode, weights in (("token_weight", [1, 0, 0]), ("sequence_balanced", [2, 0, 0]), ("sequence_balanced", [1, 1, 0])):
            with self.subTest(mode=mode, weights=weights), self.assertRaises(ValueError):
                validate_calibration_objective(rows, eos_loss_mode=mode, channelwise_loss_weight=weights)

    def test_provenance_and_boundary_fail_closed(self):
        mutations = [
            lambda row: row.pop("id"),
            lambda row: row.__setitem__("prefix_voice", "other"),
            lambda row: row.__setitem__("audio_codes", [[1, 2]]),
            lambda row: row.__setitem__("target_boundary_frame", 2),
            lambda row: row.__setitem__("prompt_sha256", "0" * 64),
            lambda row: row.__setitem__("generation_config_sha256", "0" * 64),
            lambda row: row.__setitem__("text", "不同文本"),
            lambda row: row.__setitem__("instruction", "额外条件"),
            lambda row: row["self_generated_prefix_codes"][0].__setitem__(0, 99),
            lambda row: row["ref_audio_codes"][0].__setitem__(0, 99),
            lambda row: row.pop("prefix_generator_revision"),
            lambda row: row["target_boundary_provenance"].pop("authorization"),
        ]
        for mutate in mutations:
            row = deepcopy(calibration_row())
            mutate(row)
            with self.assertRaises(ValueError):
                validate_calibration_record(row)

    def test_duplicate_ids_fail_closed(self):
        row = calibration_row()
        with self.assertRaisesRegex(ValueError, "Duplicate calibration"):
            validate_calibration_records([row, deepcopy(row)])


if __name__ == "__main__":
    unittest.main()
