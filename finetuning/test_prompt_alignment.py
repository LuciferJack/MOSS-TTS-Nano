from types import SimpleNamespace
import unittest

import torch

try:
    from .dataset import MossTTSNanoSFTDataset
except ImportError:
    from dataset import MossTTSNanoSFTDataset


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class PromptAlignmentTest(unittest.TestCase):
    @staticmethod
    def config():
        return SimpleNamespace(
            n_vq=2, im_start_token_id=101, im_end_token_id=102,
            audio_start_token_id=103, audio_end_token_id=104,
            audio_user_slot_token_id=105, audio_assistant_slot_token_id=106,
            audio_pad_token_id=0, pad_token_id=0,
        )

    def test_voice_clone_boundary_matches_inference_prompt(self):
        config = self.config()
        dataset = MossTTSNanoSFTDataset(
            [], tokenizer=CharacterTokenizer(), model_config=config, max_length=128
        )
        rows = dataset._build_prompt_rows(
            record={"text": "第一步"},
            reference_codes=[torch.tensor([[1, 2], [3, 4]])],
        )
        text_ids = rows[:, 0].tolist()
        expected_boundary = (
            [ord(char) for char in "\n</user_inst>"]
            + [config.im_end_token_id]
            + [ord("\n"), config.im_start_token_id]
            + [ord(char) for char in "assistant\n"]
            + [config.audio_start_token_id]
        )
        self.assertEqual(text_ids[-len(expected_boundary) :], expected_boundary)

    def test_collate_preserves_stable_sample_ids(self):
        dataset = MossTTSNanoSFTDataset(
            [{"id": "teacher-7", "text": "第一步", "audio_codes": [[1, 2]]}],
            tokenizer=CharacterTokenizer(), model_config=self.config(), max_length=256,
        )
        batch = dataset.collate_fn([dataset[0]])
        self.assertEqual(batch["sample_ids"], ["teacher-7"])

    def test_anonymous_sample_id_is_content_stable(self):
        record = {"text": "保护句", "audio_codes": [[1, 2]]}
        dataset = MossTTSNanoSFTDataset(
            [record, dict(record)], tokenizer=CharacterTokenizer(),
            model_config=self.config(), max_length=256,
        )
        self.assertEqual(dataset[0]["sample_id"], dataset[1]["sample_id"])
        self.assertTrue(dataset[0]["sample_id"].startswith("anon-"))


if __name__ == "__main__":
    unittest.main()
