"""Fail-closed contract for the five-row joint formula SFT pilot."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


JOINT_ROLE = "authorized_formula_joint_sft"
JOINT_ROWS = 5
ACOUSTIC_PROTECT_ROWS = 8
PILOT_AUDIO_TOTAL_WEIGHTS = (0.125, 0.5)
PILOT_N_VQ = 16
GLOBAL_LORA_TARGETS = r"^transformer\.h\.\d+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))$"


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_joint_formula_row(row: dict[str, Any]) -> bool:
    return row.get("training_role") == JOINT_ROLE


def validate_joint_formula_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if is_joint_formula_row(row)]
    if not selected:
        return []
    if len(selected) != len(rows) or len(selected) != JOINT_ROWS:
        raise ValueError("Joint formula pilot requires exactly five dedicated train rows.")
    ids: set[str] = set()
    for row in selected:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise ValueError("Joint formula pilot IDs must be non-empty and unique.")
        ids.add(sample_id)
        if row.get("domain") != "chemistry" or row.get("route") != "formula_adapter":
            raise ValueError(f"Joint row {sample_id} must be chemistry/formula_adapter scoped.")
        if row.get("codec_loss_eligible") is not True:
            raise ValueError(f"Joint row {sample_id} must explicitly enable codec loss.")
        canonical, spoken = row.get("canonical_text"), row.get("spoken_text")
        if not canonical or not spoken or row.get("text") != spoken:
            raise ValueError(f"Joint row {sample_id} must preserve canonical text and train exact spoken text.")
        codes = row.get("audio_codes")
        if not isinstance(codes, list) or not codes or not all(
            isinstance(frame, list) and len(frame) == PILOT_N_VQ for frame in codes
        ):
            raise ValueError(f"Joint row {sample_id} requires MOSS codec audio_codes.")
        if row.get("audio_codes_sha256") != _canonical_sha(codes):
            raise ValueError(f"Joint row {sample_id} audio code hash mismatch.")
        provenance = row.get("teacher_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"Joint row {sample_id} requires teacher provenance.")
        if provenance.get("authorization") != "training-and-evaluation-authorized":
            raise ValueError(f"Joint row {sample_id} teacher is not training-authorized.")
        asset_sha = provenance.get("audio_asset_sha256", "")
        if not isinstance(asset_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", asset_sha):
            raise ValueError(f"Joint row {sample_id} requires teacher asset SHA-256.")
    return selected


def validate_joint_formula_pilot(
    rows: list[dict[str, Any]], acoustic_rows: list[dict[str, Any]], *,
    enabled: bool, pcgrad: bool, protection_scope: str, behavior_rows: list[dict[str, Any]],
    eos_loss_mode: str, channel_weights: list[float], acoustic_weights: list[float],
    schedule: list[str] | None, max_train_steps: int | None, lora_rank: int,
    model_path: str = "", lora_target_modules: str = GLOBAL_LORA_TARGETS,
    lora_modules_to_save: str = "", prior_report_path: str = "",
    prior_report_sha256: str = "",
) -> None:
    selected = validate_joint_formula_rows(rows)
    if not selected:
        if enabled:
            raise ValueError("--joint-formula-pilot requires joint formula rows.")
        return
    if not enabled:
        raise ValueError("Joint formula rows require explicit --joint-formula-pilot.")
    if not pcgrad or protection_scope != "formula_scoped" or behavior_rows:
        raise ValueError("Joint formula pilot requires acoustic-only formula-scoped PCGrad.")
    if len(acoustic_rows) != ACOUSTIC_PROTECT_ROWS:
        raise ValueError("Joint formula pilot requires exactly eight acoustic protect rows.")
    if eos_loss_mode != "sequence_balanced":
        raise ValueError("Joint formula pilot requires sequence_balanced EOS loss.")
    if lora_rank <= 0:
        raise ValueError("Joint formula pilot requires a fresh LoRA over the frozen base.")
    if model_path and (Path(model_path).expanduser() / "adapter_config.json").exists():
        raise ValueError("Joint formula pilot rejects adapter stacking; provide a full frozen checkpoint.")
    if lora_target_modules != GLOBAL_LORA_TARGETS or lora_modules_to_save.strip():
        raise ValueError("Joint pilot may train only the global AR attention/MLP LoRA modules.")
    if not channel_weights or channel_weights[0] != 1.0:
        raise ValueError("Joint formula text weight must be exactly 1.")
    audio_total = sum(channel_weights[1:])
    allowed_total = next((value for value in PILOT_AUDIO_TOTAL_WEIGHTS
                          if math.isclose(audio_total, value, rel_tol=0, abs_tol=1e-12)), None)
    if any(weight <= 0 for weight in channel_weights[1:]) or allowed_total is None:
        raise ValueError("Joint pilot acoustic total must be exactly 0.125 or 0.5.")
    if len(channel_weights[1:]) != PILOT_N_VQ:
        raise ValueError("Joint formula pilot requires exactly 16 VQ loss heads.")
    expected = allowed_total / PILOT_N_VQ
    if any(not math.isclose(weight, expected, rel_tol=0, abs_tol=1e-12)
           for weight in channel_weights[1:]):
        raise ValueError("Joint pilot must weight all VQ layers evenly.")
    if allowed_total == 0.5:
        report_path = Path(prior_report_path).expanduser()
        if not report_path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", prior_report_sha256):
            raise ValueError("0.5 pilot requires the hashed prior 0.125 content-failure report.")
        actual_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
        if actual_sha != prior_report_sha256:
            raise ValueError("Prior 0.125 pilot report SHA-256 mismatch.")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Prior pilot report must be valid UTF-8 JSON.") from exc
        artifact_hashes = (report.get("candidate_weights_sha256", ""),
                           report.get("content_eval_manifest_sha256", ""))
        if (
            report.get("acoustic_total_weight") != 0.125
            or report.get("verdict") != "content_fail"
            or not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                       for value in artifact_hashes)
        ):
            raise ValueError("0.5 pilot requires a genuine 0.125 content_fail verdict.")
    elif prior_report_path or prior_report_sha256:
        raise ValueError("0.125 baseline pilot must not claim a prior failure artifact.")
    if acoustic_weights[0] != 0 or not math.isclose(sum(acoustic_weights[1:]), 1.0):
        raise ValueError("Acoustic protect objective must be 0,1.")
    expected_ids = [row["id"] for row in rows]
    if schedule != expected_ids or len(schedule) != JOINT_ROWS or max_train_steps != JOINT_ROWS:
        raise ValueError("Joint formula pilot requires one exact five-step scheduled pass.")
