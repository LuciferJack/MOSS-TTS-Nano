from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List


ROLE = "self_generated_prefix_eos_calibration"
SOURCE = "self_generated_junhao_prefix"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PROMPT_METADATA_FIELDS = (
    "instruction", "tokens", "quality", "sound_event", "ambient_sound", "language",
)


def is_calibration_record(record: Dict[str, Any]) -> bool:
    return record.get("teacher_role") == ROLE or record.get("audio_code_source") == SOURCE


def _require_sha256(record: Dict[str, Any], field: str, sample_id: str) -> None:
    value = record.get(field)
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"Calibration row {sample_id} requires 64-hex {field}.")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_calibration_record(record: Dict[str, Any]) -> None:
    sample_id = record.get("id", "<unknown>")
    if not isinstance(record.get("id"), str) or not record["id"].strip():
        raise ValueError("Calibration rows require an explicit non-empty id.")
    if record.get("teacher_role") != ROLE or record.get("audio_code_source") != SOURCE:
        raise ValueError(f"Calibration row {sample_id} has inconsistent role/source markers.")
    if record.get("prefix_voice") != "Junhao":
        raise ValueError(f"Calibration row {sample_id} must use the Junhao prefix voice.")
    if record.get("codec_loss_eligible") is not False:
        raise ValueError(f"Calibration row {sample_id} must set codec_loss_eligible=false.")
    if "audio_codes" in record:
        raise ValueError(f"Calibration row {sample_id} must not contain acoustic target audio_codes.")

    codes = record.get("self_generated_prefix_codes")
    if not isinstance(codes, list) or not codes or not all(
        isinstance(row, list) and row
        and all(isinstance(code, int) and not isinstance(code, bool) and code >= 0 for code in row)
        for row in codes
    ):
        raise ValueError(f"Calibration row {sample_id} has malformed self_generated_prefix_codes.")
    if len({len(row) for row in codes}) != 1:
        raise ValueError(f"Calibration row {sample_id} has ragged self_generated_prefix_codes.")
    boundary = record.get("target_boundary_frame")
    if not isinstance(boundary, int) or isinstance(boundary, bool) or boundary <= 0 or boundary != len(codes):
        raise ValueError(
            f"Calibration row {sample_id} target_boundary_frame must equal the generated prefix length."
        )

    revision = record.get("prefix_generator_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"Calibration row {sample_id} requires prefix_generator_revision.")
    for field in (
        "prefix_generator_model_sha256", "prefix_codes_sha256", "reference_codes_sha256",
        "prompt_sha256", "generation_config_sha256",
    ):
        _require_sha256(record, field, str(sample_id))
    prompt = record.get("generation_prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"Calibration row {sample_id} requires the exact generation_prompt.")
    if _sha256_text(prompt) != record["prompt_sha256"]:
        raise ValueError(f"Calibration row {sample_id} generation_prompt does not match prompt_sha256.")
    if prompt != record.get("text"):
        raise ValueError(f"Calibration row {sample_id} generation_prompt must exactly equal training text.")
    populated_metadata = [field for field in PROMPT_METADATA_FIELDS if record.get(field) not in (None, "")]
    if populated_metadata:
        raise ValueError(
            f"Calibration row {sample_id} cannot add prompt metadata fields: {populated_metadata}."
        )
    if _canonical_json_sha256(codes) != record["prefix_codes_sha256"]:
        raise ValueError(f"Calibration row {sample_id} prefix codes do not match prefix_codes_sha256.")
    reference_codes = record.get("ref_audio_codes")
    if not isinstance(reference_codes, list) or not reference_codes:
        raise ValueError(f"Calibration row {sample_id} requires exact Junhao ref_audio_codes.")
    if _canonical_json_sha256(reference_codes) != record["reference_codes_sha256"]:
        raise ValueError(
            f"Calibration row {sample_id} ref_audio_codes do not match reference_codes_sha256."
        )
    generation = record.get("generation_config")
    if not isinstance(generation, dict) or not generation:
        raise ValueError(f"Calibration row {sample_id} requires an explicit generation_config.")
    if _canonical_json_sha256(generation) != record["generation_config_sha256"]:
        raise ValueError(
            f"Calibration row {sample_id} generation_config does not match generation_config_sha256."
        )
    provenance = record.get("target_boundary_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Calibration row {sample_id} requires target_boundary_provenance.")
    _require_sha256(provenance, "teacher_asset_sha256", str(sample_id))
    if not str(provenance.get("provider", "")).strip() or not str(provenance.get("authorization", "")).strip():
        raise ValueError(f"Calibration row {sample_id} requires provider and authorization provenance.")


def validate_calibration_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(records)
    calibration = [row for row in rows if is_calibration_record(row)]
    seen: set[str] = set()
    for row in calibration:
        validate_calibration_record(row)
        sample_id = str(row["id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate calibration sample ID: {sample_id}")
        seen.add(sample_id)
    return calibration
