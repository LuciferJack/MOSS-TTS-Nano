"""Unit tests for finetuning.speaker_conditioner (route B1)."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from finetuning.speaker_conditioner import (
    SpeakerConditioner,
    attach_speaker_conditioner,
    embedding_dim,
    load_speaker_embeddings,
    resolve_batch_embedding,
)


class FakeBlock(nn.Module):
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                packed_metadata=None, layer_past=None, use_cache=False):
        return hidden_states, None


class FakeTransformer(nn.Module):
    def __init__(self, n_layers, hidden_size):
        super().__init__()
        self.h = nn.ModuleList([FakeBlock() for _ in range(n_layers)])
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj.weight.data = torch.eye(hidden_size)

    def forward(self, hidden_states):
        for block in self.h:
            hidden_states, _ = block(hidden_states)
        return self.proj(hidden_states)


class FakeModel(nn.Module):
    def __init__(self, n_layers, hidden_size):
        super().__init__()
        self.transformer = FakeTransformer(n_layers, hidden_size)

    def forward(self, hidden_states):
        return self.transformer(hidden_states)


class TestSpeakerConditioner(unittest.TestCase):
    def test_zero_init_is_identity(self):
        conditioner = SpeakerConditioner(embedding_dim=8, hidden_size=16, n_layers=3, film_rank=4)
        conditioner.begin_batch(torch.randn(2, 8))
        for layer in range(3):
            scale, shift = conditioner.film(layer)
            self.assertTrue(torch.all(scale == 0))
            self.assertTrue(torch.all(shift == 0))
        conditioner.end_batch()

    def test_initial_gradient_flows_to_right_factors(self):
        """At init the right factors must receive non-zero gradient (regression:
        all-zero factorization deadlocks both factors at zero gradient)."""
        torch.manual_seed(0)
        conditioner = SpeakerConditioner(embedding_dim=8, hidden_size=16, n_layers=3, film_rank=4)
        conditioner.begin_batch(torch.randn(2, 8))
        scale, shift = conditioner.film(0)
        loss = (scale.sum() + shift.sum())
        loss.backward()
        self.assertGreater(float(conditioner.scale_right.grad.abs().sum()), 0.0)
        self.assertGreater(float(conditioner.shift_right.grad.abs().sum()), 0.0)
        self.assertTrue(conditioner.encoder[0].weight.grad is not None)
        conditioner.end_batch()

    def test_dims_validation(self):
        with self.assertRaises(ValueError):
            SpeakerConditioner(embedding_dim=0, hidden_size=16, n_layers=3, film_rank=4)

    def test_begin_batch_shape_check(self):
        conditioner = SpeakerConditioner(embedding_dim=8, hidden_size=16, n_layers=3, film_rank=4)
        with self.assertRaises(ValueError):
            conditioner.begin_batch(torch.randn(2, 7))

    def test_film_requires_begin_batch(self):
        conditioner = SpeakerConditioner(embedding_dim=8, hidden_size=16, n_layers=3, film_rank=4)
        with self.assertRaises(RuntimeError):
            conditioner.film(0)

    def test_attach_and_modify(self):
        torch.manual_seed(0)
        model = FakeModel(n_layers=3, hidden_size=16)
        conditioner = SpeakerConditioner(embedding_dim=8, hidden_size=16, n_layers=3, film_rank=4)
        with torch.no_grad():
            conditioner.scale_left += 0.1
            conditioner.scale_right += 0.1
            conditioner.shift_left += 0.1
            conditioner.shift_right += 0.1
        handles = attach_speaker_conditioner(model, conditioner)
        self.assertEqual(len(handles), 3)
        x = torch.randn(2, 5, 16)
        conditioner.begin_batch(torch.randn(2, 8))
        out = model(x)
        conditioner.end_batch()
        # hooks active: every block input modified -> output differs from unconditioned
        out_plain = FakeModel(3, 16)
        out_plain.transformer.proj.weight.data = torch.eye(16)
        self.assertFalse(torch.allclose(out, out_plain(x), atol=1e-6))
        # without begin_batch the hook is a pass-through
        out2 = model(x)
        self.assertTrue(torch.allclose(out2, out_plain(x), atol=1e-6))
        for handle in handles:
            handle.remove()

    def test_layer_count_mismatch(self):
        model = FakeModel(n_layers=2, hidden_size=16)
        conditioner = SpeakerConditioner(embedding_dim=8, hidden_size=16, n_layers=3, film_rank=4)
        with self.assertRaises(ValueError):
            attach_speaker_conditioner(model, conditioner)


class TestLoadEmbeddings(unittest.TestCase):
    def _write(self, references):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "speaker-embeddings.json"
        path.write_text(json.dumps({"references": references}), encoding="utf-8")
        self.addCleanup(tmp.cleanup)
        return str(path)

    @staticmethod
    def _unit(dims):
        v = [0.0] * dims
        v[0] = 1.0
        return v

    def test_load_ok(self):
        path = self._write({
            "junhao_real_a03": {"embedding": self._unit(8)},
            "junhao_real_a01": {"embedding": self._unit(8)},
        })
        embeddings = load_speaker_embeddings(path)
        self.assertEqual(embedding_dim(embeddings), 8)
        self.assertEqual(sorted(embeddings), ["junhao_real_a01", "junhao_real_a03"])

    def test_rejects_non_normalized(self):
        path = self._write({"ref": {"embedding": [2.0, 0.0]}})
        with self.assertRaisesRegex(ValueError, "L2"):
            load_speaker_embeddings(path)

    def test_rejects_non_finite(self):
        path = self._write({"ref": {"embedding": [float("nan"), 0.0]}})
        with self.assertRaises(ValueError):
            load_speaker_embeddings(path)

    def test_rejects_dim_drift(self):
        path = self._write({
            "a": {"embedding": self._unit(8)},
            "b": {"embedding": self._unit(9)},
        })
        with self.assertRaisesRegex(ValueError, "dimension"):
            load_speaker_embeddings(path)

    def test_rejects_empty(self):
        path = self._write({})
        with self.assertRaises(ValueError):
            load_speaker_embeddings(path)


class TestResolveBatch(unittest.TestCase):
    def test_resolve_stacks(self):
        embeddings = {"ref_a": torch.tensor([1.0, 0.0]), "ref_b": torch.tensor([0.0, 1.0])}
        ref_map = {"s1": "ref_a", "s2": "ref_b"}
        batch = resolve_batch_embedding(["s1", "s2"], reference_id_by_sample=ref_map, embeddings=embeddings)
        self.assertTrue(torch.equal(batch, torch.tensor([[1.0, 0.0], [0.0, 1.0]])))

    def test_missing_sample(self):
        with self.assertRaisesRegex(ValueError, "no reference provenance"):
            resolve_batch_embedding(["ghost"], reference_id_by_sample={}, embeddings={})

    def test_missing_reference(self):
        with self.assertRaisesRegex(ValueError, "missing from the speaker embeddings"):
            resolve_batch_embedding(
                ["s1"], reference_id_by_sample={"s1": "ref_x"},
                embeddings={"ref_y": torch.tensor([1.0, 0.0])},
            )


if __name__ == "__main__":
    unittest.main()
