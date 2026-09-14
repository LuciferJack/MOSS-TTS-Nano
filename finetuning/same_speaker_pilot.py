"""Fail-closed contract for the one-train/one-heldout Junhao pilot.

This module intentionally does not start training.  It validates the immutable
bundle that must pass before the experiment is wired into ``sft.py``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


N_VQ = 16
TRAIN_ROLE = "same_speaker_professional_train"
HELDOUT_ROLE = "same_speaker_professional_heldout"
PROTECT_ROLE = "same_speaker_natural_protector"
GLOBAL_LORA_TARGETS = r"^transformer\.h\.\d+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))$"
ALLOWED_STEPS = (1, 2, 4)
TEXT_WEIGHT = 1.0
AUDIO_TOTAL_WEIGHT = 0.125
TRAIN_ID = "junhao_caoh"
HELDOUT_ID = "junhao_hydrate"
TRAIN_TEXT = "西 诶 左括号 欧 诶吃 右括号 二，氢氧化钙"
HELDOUT_TEXT = "C O C L 二 点 六 H 二 欧，是六水合氯化钴。"
PREFLIGHT_SHA256 = "dc8207757d9662a14ce64dfe533b0d0e434512133493d9541219004a598874fb"
SOURCE_MANIFEST_SHA256 = "348d595cc33f845318ad7c4b3e180c76bd1fb720139f8006186fabb7369a52d5"
SOURCE_GATE_SHAS = {
    TRAIN_ID: "0d55e9438a549b54aed9ae3a6abb9e38627443802bede5b7a99525ff2400507e",
    HELDOUT_ID: "306193f12c3328b53f5215164e86fab388e9689d6dbdc0e604c1869cca19757f",
}
HUMAN_LEDGER_SHA256 = "1365fd92327ebc38748e3218688ff2b13cc32572d3a4283ad12ca349461b2aab"
BASE_MODEL_SHA256 = "24003f2f11ac8a2cbf70514db2d8f1c02fb451aa6b3c0bffc9da09f31cd7caa5"
REFERENCE_ID = "junhao_real_a03"


def canonical_sha(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA-256.")
    return value


def _load_preflight(path: str, expected_sha: str) -> dict[str, Any]:
    gate_path = Path(path).expanduser()
    if not gate_path.is_file():
        raise ValueError("Same-speaker machine preflight manifest is missing.")
    actual = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    if expected_sha != PREFLIGHT_SHA256 or actual != expected_sha:
        raise ValueError("Same-speaker preflight manifest SHA-256 mismatch.")
    try:
        report = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Preflight manifest must be valid UTF-8 JSON.") from exc
    if report.get("schema") != "min-same-voice-causal-pilot-preflight-v1":
        raise ValueError("Unexpected same-speaker preflight schema.")
    return report


def _validate_reference(row: dict[str, Any]) -> tuple[str, str]:
    sample_id = row.get("id", "<unknown>")
    codes = row.get("ref_audio_codes")
    if not isinstance(codes, list) or not codes or not all(
        isinstance(frame, list) and len(frame) == N_VQ for frame in codes
    ):
        raise ValueError(f"{sample_id} requires one explicit 16-VQ Junhao reference.")
    code_sha = canonical_sha(codes)
    if row.get("reference_codes_sha256") != code_sha:
        raise ValueError(f"{sample_id} reference code SHA-256 mismatch.")
    provenance = row.get("reference_provenance")
    if (not isinstance(provenance, dict) or provenance.get("speaker") != "Junhao"
            or provenance.get("id") != REFERENCE_ID):
        raise ValueError(f"{sample_id} reference must have Junhao provenance.")
    return code_sha, _sha(provenance.get("audio_asset_sha256"), "reference audio_asset_sha256")


def validate_same_speaker_pilot(
    train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]],
    protect_rows: list[dict[str, Any]], *, preflight_manifest_path: str,
    preflight_manifest_sha256: str, max_train_steps: int, eos_loss_mode: str,
    channel_weights: list[float], protect_weights: list[float], pcgrad: bool,
    lora_rank: int, model_path: str, lora_target_modules: str = GLOBAL_LORA_TARGETS,
    lora_modules_to_save: str = "", tail_weighting: bool = False,
) -> dict[str, Any]:
    """Validate the causal pilot and return a hashable resolved contract."""
    if len(train_rows) != 1 or train_rows[0].get("training_role") != TRAIN_ROLE:
        raise ValueError("Pilot requires exactly one professional Junhao train row.")
    if len(heldout_rows) != 1 or heldout_rows[0].get("training_role") != HELDOUT_ROLE:
        raise ValueError("Pilot requires exactly one professional Junhao heldout row.")
    if len(protect_rows) != 5 or any(row.get("protector_role") != PROTECT_ROLE for row in protect_rows):
        raise ValueError("Pilot requires exactly five natural Junhao protectors.")
    train, heldout = train_rows[0], heldout_rows[0]
    if train.get("id") != TRAIN_ID or train.get("category") != "professional":
        raise ValueError(f"Pilot train row must be pinned to {TRAIN_ID}.")
    if heldout.get("id") != HELDOUT_ID or heldout.get("category") != "professional":
        raise ValueError(f"Pilot heldout row must be pinned to {HELDOUT_ID}.")
    if train.get("text") != TRAIN_TEXT or heldout.get("text") != HELDOUT_TEXT:
        raise ValueError("Pilot professional spoken targets do not match the frozen experiment.")
    all_rows = train_rows + heldout_rows + protect_rows
    ids = [row.get("id") for row in all_rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("All pilot IDs must be non-empty and globally unique.")
    if heldout.get("gradient_eligible") is not False:
        raise ValueError("Heldout must explicitly declare gradient_eligible=false.")
    if any(row.get("gradient_eligible") is not True for row in train_rows + protect_rows):
        raise ValueError("Only train/protector rows may explicitly enter gradients.")
    structures = (train.get("structure_signature"), heldout.get("structure_signature"))
    if any(not isinstance(value, str) or not value for value in structures):
        raise ValueError("Both professional rows require non-empty structure signatures.")
    if structures[0] == structures[1]:
        raise ValueError("Heldout must use a different professional structure.")
    if train.get("text") == heldout.get("text"):
        raise ValueError("Heldout text leaked from train.")
    if any(row.get("category") != "natural" for row in protect_rows):
        raise ValueError("Professional examples are forbidden from the protector set.")

    reference_pairs = {_validate_reference(row) for row in all_rows}
    if len(reference_pairs) != 1:
        raise ValueError("Train/eval/protection must use the exact same explicit reference.")
    reference_code_sha, reference_audio_sha = next(iter(reference_pairs))

    target_shas: set[str] = set()
    for row in all_rows:
        if row.get("speaker") != "Junhao":
            raise ValueError(f"{row['id']} target speaker must be Junhao.")
        codes = row.get("audio_codes")
        if not isinstance(codes, list) or not codes or not all(
            isinstance(frame, list) and len(frame) == N_VQ for frame in codes
        ):
            raise ValueError(f"{row['id']} requires 16-VQ target audio codes.")
        if row.get("audio_codes_sha256") != canonical_sha(codes):
            raise ValueError(f"{row['id']} target audio code SHA-256 mismatch.")
        target_shas.add(_sha(row.get("audio_asset_sha256"), "target audio_asset_sha256"))
    if len(target_shas) != len(all_rows):
        raise ValueError("Target audio assets must be unique; heldout/protector leakage detected.")
    if reference_audio_sha in target_shas:
        raise ValueError("Reference audio must be independent of every target/protector utterance.")

    report = _load_preflight(preflight_manifest_path, preflight_manifest_sha256)
    if report.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise ValueError("Preflight source manifest SHA-256 mismatch.")
    assets = {item.get("id"): item for item in report.get("assets", []) if isinstance(item, dict)}
    for row in (train, heldout):
        asset = assets.get(row["id"])
        if (not asset or asset.get("audio_sha256") != row["audio_asset_sha256"]
                or asset.get("codec_codes_canonical_json_sha256") != row["audio_codes_sha256"]
                or asset.get("source_gate_artifact_sha256") != SOURCE_GATE_SHAS[row["id"]]
                or asset.get("human_gate_ledger_sha256") != HUMAN_LEDGER_SHA256
                or asset.get("reference", {}).get("audio_sha256") != reference_audio_sha
                or asset.get("reference", {}).get("codec_codes_canonical_json_sha256") != reference_code_sha):
            raise ValueError(f"{row['id']} does not bind the reviewed machine artifacts/ledger.")
    natural = {item.get("id"): item for item in report.get("natural_pcgrad_only", []) if isinstance(item, dict)}
    for row in protect_rows:
        item = natural.get(row["id"])
        if not item or item.get("audio_sha256") != row["audio_asset_sha256"] or item.get("codec_codes_sha256") != row["audio_codes_sha256"]:
            raise ValueError(f"{row['id']} does not bind the reviewed natural protector artifact.")

    if max_train_steps not in ALLOWED_STEPS:
        raise ValueError("Pilot may stop only at 1, 2, or 4 optimizer steps.")
    if eos_loss_mode != "sequence_balanced" or tail_weighting:
        raise ValueError("Pilot requires sequence-balanced EOS and forbids failed tail weighting.")
    if len(channel_weights) != N_VQ + 1 or channel_weights[0] != TEXT_WEIGHT:
        raise ValueError("Pilot text weight must be 1 with exactly 16 VQ weights.")
    expected = AUDIO_TOTAL_WEIGHT / N_VQ
    if any(not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)
           for value in channel_weights[1:]):
        raise ValueError("Pilot audio total must be 0.125, evenly split across 16 VQ heads.")
    expected_protect = 1.0 / N_VQ
    if (not pcgrad or len(protect_weights) != N_VQ + 1 or protect_weights[0] != 0
            or any(not math.isfinite(value) or not math.isclose(
                value, expected_protect, rel_tol=0, abs_tol=1e-12
            ) for value in protect_weights[1:])):
        raise ValueError("Five natural protectors require acoustic-only PCGrad weights 0,1.")
    if lora_rank <= 0 or lora_target_modules != GLOBAL_LORA_TARGETS or lora_modules_to_save.strip():
        raise ValueError("Pilot requires fresh global attention+MLP LoRA only.")
    model_root = Path(model_path).expanduser()
    if not model_root.is_dir():
        raise ValueError("Pilot requires an existing clean full checkpoint directory.")
    if (model_root / "adapter_config.json").exists():
        raise ValueError("Pilot rejects adapter stacking; provide a clean full checkpoint.")
    weights_path = model_root / "pytorch_model.bin"
    if not weights_path.is_file() or _file_sha256(weights_path) != BASE_MODEL_SHA256:
        raise ValueError("Pilot model weights do not match the approved frozen clean base.")

    return {
        "schema": "same-speaker-single-train-pilot-v1",
        "train_id": train["id"], "heldout_id": heldout["id"],
        "protector_ids": [row["id"] for row in protect_rows],
        "reference_codes_sha256": reference_code_sha,
        "reference_audio_sha256": reference_audio_sha,
        "reference_id": REFERENCE_ID,
        "preflight_manifest_sha256": preflight_manifest_sha256,
        "candidate_steps": [step for step in ALLOWED_STEPS if step <= max_train_steps],
        "audio_total_weight": AUDIO_TOTAL_WEIGHT,
    }
