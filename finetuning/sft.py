from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.dataclasses import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.utils import cached_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.common import format_duration, format_timestamp, load_jsonl_spec
from finetuning.content_supervision import (
    ContentHead,
    alignment_vocab,
    build_content_targets,
    compute_content_loss,
    load_content_alignment,
    validate_alignment_covers_records,
)
from finetuning.dataset import MossTTSNanoSFTDataset, stable_sample_id
from finetuning.eos_calibration import validate_calibration_records
from finetuning.joint_formula_sft import validate_joint_formula_pilot
from finetuning.same_speaker_pilot import validate_same_speaker_pilot
from finetuning.protected_gradient import is_feasible_dot, project_teacher_gradient
from finetuning.speaker_conditioner import (
    SpeakerConditioner,
    attach_speaker_conditioner,
    embedding_dim,
    load_speaker_embeddings,
    resolve_batch_embedding,
)

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "MOSS-TTS-Nano"
DEFAULT_CODEC_PATH = REPO_ROOT / "models" / "MOSS-Audio-Tokenizer-Nano"

SCHEDULER_CHOICES = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
)
EOS_LOSS_MODES = ("token_weight", "sequence_balanced")

MODEL_SUPPORT_FILES = (
    "__init__.py",
    "configuration_moss_tts_nano.py",
    "gpt2_decoder.py",
    "modeling_moss_tts_nano.py",
    "prompting.py",
    "tokenization_moss_tts_nano.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple supervised finetuning for MOSS-TTS-Nano.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--codec-path", type=str, default=str(DEFAULT_CODEC_PATH))
    parser.add_argument(
        "--train-jsonl",
        type=str,
        required=True,
        help="A single JSONL, directory, glob, or comma-separated list of JSONLs produced by prepare_data.py.",
    )
    parser.add_argument("--output-dir", type=str, default="output/moss_tts_nano_sft")
    parser.add_argument("--max-length", type=int, default=1024, help="Fixed full sequence length before shift.")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear", choices=SCHEDULER_CHOICES)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--diagnostics-jsonl",
        type=str,
        default="",
        help="Optional JSONL path for per-channel losses and pre-clip module gradient norms.",
    )
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--train-schedule-json",
        type=str,
        default="",
        help=(
            "Optional JSON array containing every training sample ID exactly once. "
            "When set, training is sequential and max_train_steps must equal its length."
        ),
    )
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow bounded CPU diagnostics. Full training remains CUDA-first.",
    )
    parser.add_argument("--attn-implementation", type=str, default="auto")
    parser.add_argument(
        "--channelwise-loss-weight",
        type=str,
        default="1,32",
        help=(
            "Either n_heads values (text,vq0,...,vqN) or 2 values (text_weight,total_audio_weight). "
            "The total audio weight will be evenly split across all audio heads."
        ),
    )
    parser.add_argument(
        "--protect-channelwise-loss-weight", type=str, default="0,1",
        help="Loss weights for the PCGrad preservation batch; defaults to acoustic-only protection.",
    )
    parser.add_argument(
        "--eos-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Relative weight for the single audio-end target in the text channel. "
            "The default 1.0 is exactly the legacy mean cross-entropy objective."
        ),
    )
    parser.add_argument(
        "--eos-loss-mode",
        choices=EOS_LOSS_MODES,
        default="token_weight",
        help=(
            "token_weight preserves token-mean CE; sequence_balanced gives each sample's "
            "continue mean and stop loss explicit, length-independent weights."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=0, help="Enable LoRA when greater than zero.")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=r"^transformer\.h\.\d+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))$",
        help="PEFT target-module regex. Defaults to the 12-layer global AR transformer only.",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        default="",
        help="Comma-separated non-LoRA modules to train and store with the adapter.",
    )
    parser.add_argument("--pcgrad", action="store_true", help="Project teacher gradients against a preservation batch.")
    parser.add_argument("--protect-jsonl", type=str, default="", help="Held-out preservation JSONL required by --pcgrad.")
    parser.add_argument(
        "--behavior-protect-jsonl", type=str, default="",
        help="Two pinned text/EOS behavior rows used only as PCGrad constraints.",
    )
    parser.add_argument(
        "--behavior-protect-channelwise-loss-weight", type=str, default="1,0",
        help="Behavior constraint weights; calibration training requires exactly text-only 1,0.",
    )
    parser.add_argument(
        "--calibration-protection-scope", choices=("dual", "formula_scoped"), default="dual",
        help="formula_scoped retains eight acoustic constraints but excludes behavior replay.",
    )
    parser.add_argument(
        "--calibration-source-model-sha256", default="",
        help="Expected merged policy SHA-256; mandatory for round-2 calibration.",
    )
    parser.add_argument(
        "--calibration-source-revision", default="",
        help="Expected immutable generator revision; mandatory for round-2 calibration.",
    )
    parser.add_argument(
        "--joint-formula-pilot", action="store_true",
        help="Enable the fail-closed five-row joint text+codec formula pilot.",
    )
    parser.add_argument("--joint-formula-prior-report", default="")
    parser.add_argument("--joint-formula-prior-report-sha256", default="")
    parser.add_argument(
        "--joint-formula-tail-weighting", action="store_true",
        help="Enable the v3 body:tail acoustic frame weighting of 1:2 without changing total audio weight.",
    )
    parser.add_argument("--joint-formula-v2-fail-report", default="")
    parser.add_argument("--joint-formula-v2-fail-report-sha256", default="")
    parser.add_argument("--joint-formula-trace-audit", default="")
    parser.add_argument("--joint-formula-trace-audit-sha256", default="")
    parser.add_argument("--joint-formula-alignment-report", default="")
    parser.add_argument("--joint-formula-alignment-report-sha256", default="")
    parser.add_argument("--same-speaker-pilot", action="store_true")
    parser.add_argument("--same-speaker-heldout-jsonl", default="")
    parser.add_argument("--same-speaker-preflight-manifest", default="")
    parser.add_argument("--same-speaker-preflight-manifest-sha256", default="")
    parser.add_argument(
        "--content-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the auxiliary content-head CE on aligned response frames (route A1). "
            "The default 0.0 preserves the legacy objective bit-for-bit."
        ),
    )
    parser.add_argument(
        "--content-alignment-jsonl",
        default="",
        help="Per-frame character alignment manifest; required when --content-loss-weight > 0.",
    )
    parser.add_argument(
        "--speaker-conditioning",
        action="store_true",
        help="Enable the independent CAMPPlus-FiLM speaker conditioner (route B1).",
    )
    parser.add_argument(
        "--speaker-embeddings-jsonl",
        default="",
        help="Pinned speaker-embeddings manifest; required when --speaker-conditioning is set.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_length <= 8:
        raise ValueError("`max_length` must be > 8.")
    if args.per_device_batch_size <= 0:
        raise ValueError("`per_device_batch_size` must be > 0.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("`gradient_accumulation_steps` must be > 0.")
    if args.learning_rate <= 0:
        raise ValueError("`learning_rate` must be > 0.")
    if args.weight_decay < 0:
        raise ValueError("`weight_decay` must be >= 0.")
    if args.warmup_steps < 0:
        raise ValueError("`warmup_steps` must be >= 0.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("`warmup_ratio` must be in [0, 1).")
    if args.num_epochs <= 0:
        raise ValueError("`num_epochs` must be > 0.")
    if args.content_loss_weight < 0:
        raise ValueError("`content_loss_weight` must be >= 0.")
    if args.content_loss_weight > 0 and not args.content_alignment_jsonl:
        raise ValueError("--content-alignment-jsonl is required when --content-loss-weight > 0.")
    if args.content_alignment_jsonl and args.content_loss_weight <= 0:
        raise ValueError("--content-alignment-jsonl requires --content-loss-weight > 0.")
    if args.speaker_conditioning and not args.speaker_embeddings_jsonl:
        raise ValueError("--speaker-embeddings-jsonl is required when --speaker-conditioning is set.")
    if args.speaker_embeddings_jsonl and not args.speaker_conditioning:
        raise ValueError("--speaker-embeddings-jsonl requires --speaker-conditioning.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("`max_train_steps` must be > 0 when set.")
    if args.max_grad_norm < 0:
        raise ValueError("`max_grad_norm` must be >= 0.")
    if args.logging_steps <= 0:
        raise ValueError("`logging_steps` must be > 0.")
    if args.save_every_epochs <= 0:
        raise ValueError("`save_every_epochs` must be > 0.")
    if args.num_workers < 0:
        raise ValueError("`num_workers` must be >= 0.")
    if args.lora_rank < 0:
        raise ValueError("`lora_rank` must be >= 0.")
    if args.lora_alpha <= 0:
        raise ValueError("`lora_alpha` must be > 0.")
    if not math.isfinite(args.eos_loss_weight) or args.eos_loss_weight <= 0:
        raise ValueError("`eos_loss_weight` must be finite and > 0.")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("`lora_dropout` must be in [0, 1).")
    if args.pcgrad and not args.protect_jsonl:
        raise ValueError("`--protect-jsonl` is required with `--pcgrad`.")
    if args.pcgrad and args.gradient_accumulation_steps != 1:
        raise ValueError("PCGrad currently requires `--gradient-accumulation-steps 1`.")
    if args.behavior_protect_jsonl and not args.pcgrad:
        raise ValueError("`--behavior-protect-jsonl` requires `--pcgrad`.")
    if args.train_schedule_json and args.per_device_batch_size != 1:
        raise ValueError("Scheduled training requires `--per-device-batch-size 1` for step-level auditability.")
    if args.train_schedule_json and args.gradient_accumulation_steps != 1:
        raise ValueError("Scheduled training requires `--gradient-accumulation-steps 1`.")
    same_speaker_fields = (args.same_speaker_heldout_jsonl, args.same_speaker_preflight_manifest,
                           args.same_speaker_preflight_manifest_sha256)
    if (args.same_speaker_pilot and not all(bool(value) for value in same_speaker_fields)) or (
        not args.same_speaker_pilot and any(bool(value) for value in same_speaker_fields)
    ):
        raise ValueError("Same-speaker pilot requires its heldout JSONL and preflight path/SHA together.")
    if args.same_speaker_pilot and args.joint_formula_pilot:
        raise ValueError("Same-speaker and joint-formula pilots are mutually exclusive.")
    if args.same_speaker_pilot:
        if args.per_device_batch_size != 1:
            raise ValueError("Same-speaker pilot requires batch size 1 for per-protector PCGrad.")
        if args.save_every_epochs != 1:
            raise ValueError("Same-speaker pilot requires one checkpoint per epoch for step 1/2/4 gates.")
        if args.max_train_steps is None or args.num_epochs < args.max_train_steps:
            raise ValueError("Same-speaker singleton pilot requires num_epochs >= max_train_steps.")


def pcgrad_backward(
    *, accelerator, model, teacher_loss: torch.Tensor,
    protector_loss: Optional[torch.Tensor] = None,
    protector_losses: Optional[List[torch.Tensor]] = None,
    protector_loss_factories: Optional[List[Callable[[], torch.Tensor]]] = None,
):
    """Apply only the teacher update, projected to avoid harming the protector.

    The protector gradient defines a half-space constraint; it is deliberately
    not added to the optimizer gradient.  Adding it would actively fit the
    preservation batch on every step and can distort duration/EOS behaviour.
    """
    losses = list(protector_losses or ([] if protector_loss is None else [protector_loss]))
    factories = list(protector_loss_factories or [])
    if losses and factories:
        raise ValueError("Pass protector losses or factories, not both.")
    if not losses and not factories:
        raise ValueError("PCGrad requires at least one protector loss.")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_grads = [None if p.grad is None else p.grad.detach().clone() for p in trainable]
    for parameter in trainable:
        parameter.grad = None
    # Factory mode is used by dual protection so only one protector forward
    # graph exists at a time. Keep the constraint vectors on CPU to avoid ten
    # model-sized gradient copies consuming accelerator memory.
    if factories:
        accelerator.backward(teacher_loss)
    else:
        all_protector_grads = []
        for loss in losses:
            accelerator.backward(loss)
            all_protector_grads.append(
                [None if p.grad is None else p.grad.detach().clone() for p in trainable]
            )
            for parameter in trainable:
                parameter.grad = None
        accelerator.backward(teacher_loss)
    teacher_grads = [None if p.grad is None else p.grad.detach().clone() for p in trainable]
    active = [i for i, teacher in enumerate(teacher_grads) if teacher is not None]
    if not active:
        raise RuntimeError("PCGrad found no teacher gradient.")
    teacher_flat = torch.cat([teacher_grads[i].float().reshape(-1).cpu() for i in active])
    protector_flats: List[torch.Tensor] = []
    if factories:
        for parameter in trainable:
            parameter.grad = None
        for factory in factories:
            accelerator.backward(factory())
            protector_flats.append(torch.cat([
                (torch.zeros_like(teacher_grads[i]) if trainable[i].grad is None else trainable[i].grad)
                .detach().float().reshape(-1).cpu()
                for i in active
            ]))
            for parameter in trainable:
                parameter.grad = None
    else:
        for grads in all_protector_grads:
            protector_flats.append(torch.cat([
                (torch.zeros_like(teacher_grads[i]) if grads[i] is None else grads[i])
                .float().reshape(-1).cpu()
                for i in active
            ]))
    projected, report = project_teacher_gradient(teacher_flat, protector_flats)
    active_set = set(active)
    offset = 0
    for i, parameter in enumerate(trainable):
        previous = optimizer_grads[i]
        teacher = teacher_grads[i]
        if i in active_set:
            count = teacher.numel()
            resolved = projected[offset:offset + count].reshape_as(teacher)
            offset += count
            parameter.grad = resolved.to(device=parameter.device, dtype=parameter.dtype)
        elif teacher is not None:
            parameter.grad = teacher
        if previous is not None:
            parameter.grad = previous if parameter.grad is None else parameter.grad + previous
    return report


def configure_torch_backends() -> None:
    if not torch.cuda.is_available():
        return
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def resolve_torch_dtype(mixed_precision: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_accelerate_mixed_precision(mixed_precision: str) -> str:
    if not torch.cuda.is_available():
        return "no"
    return mixed_precision


def resolve_attn_implementation(requested: str, dtype: torch.dtype) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "eager"
    if dtype in {torch.float16, torch.bfloat16}:
        try:
            import flash_attn  # noqa: F401

            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return "flash_attention_2"
        except Exception:
            pass
    return "sdpa"


def resolve_warmup_steps(args: argparse.Namespace, num_training_steps: int) -> int:
    if args.warmup_steps > 0:
        return args.warmup_steps
    if args.warmup_ratio > 0:
        return math.ceil(num_training_steps * args.warmup_ratio)
    return 0


def parse_channelwise_loss_weight(spec: str, n_heads: int) -> List[float]:
    values = [float(item.strip()) for item in str(spec).split(",") if item.strip()]
    if len(values) == n_heads:
        resolved = values
    elif len(values) == 2 and n_heads > 1:
        text_weight, total_audio_weight = values
        per_audio_weight = total_audio_weight / float(n_heads - 1)
        resolved = [text_weight] + [per_audio_weight] * (n_heads - 1)
    else:
        raise ValueError(
            f"`channelwise_loss_weight` expects either {n_heads} values or 2 values, got {len(values)}."
        )
    if sum(resolved) <= 0:
        raise ValueError("`channelwise_loss_weight` must sum to a positive value.")
    return resolved


def resolve_objective_loss_weights(
    args: argparse.Namespace,
    n_heads: int,
) -> tuple[List[float], List[float]]:
    """Resolve teacher and protector objectives independently.

    Keeping this mapping in one tested helper prevents the preservation batch
    from accidentally inheriting the teacher's text/EOS-only objective.
    """
    teacher_weights = parse_channelwise_loss_weight(args.channelwise_loss_weight, n_heads)
    protector_weights = parse_channelwise_loss_weight(args.protect_channelwise_loss_weight, n_heads)
    return teacher_weights, protector_weights


def validate_calibration_objective(
    records: List[Dict[str, Any]], *, eos_loss_mode: str, channelwise_loss_weight: List[float]
) -> None:
    calibration = validate_calibration_records(records)
    if not calibration:
        return
    if eos_loss_mode != "sequence_balanced":
        raise ValueError("Self-generated EOS calibration requires sequence_balanced EOS loss.")
    if channelwise_loss_weight[0] != 1 or any(weight != 0 for weight in channelwise_loss_weight[1:]):
        raise ValueError("Self-generated EOS calibration requires channel weights 1,0 (zero VQ loss).")


def validate_calibration_protection_enabled(records: List[Dict[str, Any]], *, pcgrad: bool) -> None:
    if validate_calibration_records(records) and not pcgrad:
        raise ValueError("On-policy EOS calibration requires --pcgrad and acoustic protection.")


def validate_behavior_protection(
    records: List[Dict[str, Any]], acoustic_records: List[Dict[str, Any]],
    behavior_records: List[Dict[str, Any]],
    *, acoustic_weights: List[float], behavior_weights: List[float],
    eos_loss_mode: str, train_schedule: Optional[List[str]] = None,
    protection_scope: str = "dual", expected_source_sha256: str = "",
    expected_source_revision: str = "",
) -> None:
    calibration = validate_calibration_records(records)
    if not calibration:
        if behavior_records:
            raise ValueError("Behavior protection is only supported for EOS calibration training.")
        return
    if protection_scope not in {"dual", "formula_scoped"}:
        raise ValueError(f"Unknown calibration protection scope: {protection_scope}")
    if protection_scope == "dual" and not behavior_records:
        raise ValueError("EOS calibration requires --behavior-protect-jsonl.")
    if len(calibration) != len(records):
        raise ValueError("Behavior replay must not be mixed into calibration train rows.")
    if len(calibration) != 5:
        raise ValueError("EOS calibration schedule requires exactly five calibration train rows.")
    if train_schedule is not None and (
        len(train_schedule) != 5 or train_schedule != [stable_sample_id(row) for row in records]
    ):
        raise ValueError("EOS calibration requires an explicit five-row schedule matching train order.")
    if len(acoustic_records) != 8:
        raise ValueError("Dual PCGrad requires exactly eight acoustic protector rows.")
    if protection_scope == "dual" and len(behavior_records) != 2:
        raise ValueError("Behavior protection requires exactly two pinned replay rows.")
    if protection_scope == "formula_scoped" and behavior_records:
        raise ValueError("Formula-scoped calibration must not optimize against behavior replay.")
    if validate_calibration_records(behavior_records):
        raise ValueError("Behavior protect rows must be ordinary replay rows, not calibration rows.")
    train_ids = {stable_sample_id(row) for row in records}
    behavior_ids = [stable_sample_id(row) for row in behavior_records]
    if len(set(behavior_ids)) != len(behavior_ids) or train_ids.intersection(behavior_ids):
        raise ValueError("Behavior protector IDs must be unique and disjoint from train rows.")
    if eos_loss_mode != "sequence_balanced":
        raise ValueError("Behavior protection requires sequence_balanced EOS loss.")
    if acoustic_weights[0] != 0 or not math.isclose(sum(acoustic_weights[1:]), 1.0):
        raise ValueError("Acoustic protection requires channel weights 0,1.")
    if protection_scope == "dual" and (
        behavior_weights[0] != 1 or any(weight != 0 for weight in behavior_weights[1:])
    ):
        raise ValueError("Behavior protection requires channel weights 1,0 (zero VQ loss).")
    rounds = {row.get("on_policy_round", 1) for row in calibration}
    source_shas = {row["prefix_generator_model_sha256"] for row in calibration}
    if len(rounds) == 1 and next(iter(rounds)) >= 2:
        if not expected_source_sha256 or source_shas != {expected_source_sha256}:
            raise ValueError("Round-2 calibration source SHA must match the expected merged candidate SHA.")
        source_revisions = {row["prefix_generator_revision"] for row in calibration}
        if not expected_source_revision or source_revisions != {expected_source_revision}:
            raise ValueError("Round-2 calibration revision must match the expected merged candidate revision.")


def validate_round2_model_baseline(
    records: List[Dict[str, Any]], model_path: str, *, expected_sha256: str, lora_rank: int,
) -> None:
    """Bind DAgger round-2 rows to a merged checkpoint and a fresh LoRA."""
    calibration = validate_calibration_records(records)
    if not calibration or min(row.get("on_policy_round", 1) for row in calibration) < 2:
        return
    if lora_rank <= 0:
        raise ValueError("Round-2 calibration requires a fresh LoRA on the merged parent checkpoint.")
    root = Path(model_path).expanduser().resolve()
    if (root / "adapter_config.json").exists():
        raise ValueError("Round-2 model path must be a merged parent, not a stacked LoRA adapter.")
    weights = root / "pytorch_model.bin"
    if not weights.is_file():
        raise ValueError("Round-2 merged parent must contain authoritative pytorch_model.bin weights.")
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if not expected_sha256 or actual != expected_sha256:
        raise ValueError(
            f"Round-2 model SHA mismatch: expected {expected_sha256 or '<missing>'}, actual {actual}."
        )


def build_optimizer(model, args: argparse.Namespace, extra_modules=()) -> AdamW:
    parameters = list(model.parameters())
    for module in extra_modules:
        parameters.extend(module.parameters())
    return AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )


def apply_train_schedule(
    records: List[Dict[str, Any]], schedule_path: str,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Validate and apply a single auditable epoch schedule."""
    if not schedule_path:
        return records, []
    path = Path(schedule_path)
    try:
        schedule = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read train schedule {path}: {exc}") from exc
    if not isinstance(schedule, list) or not schedule or not all(isinstance(item, str) and item for item in schedule):
        raise ValueError("Train schedule must be a non-empty JSON array of sample ID strings.")
    if len(schedule) != len(set(schedule)):
        raise ValueError("Train schedule contains duplicate sample IDs.")

    by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        sample_id = stable_sample_id(record)
        if sample_id in by_id:
            raise ValueError(f"Training records contain duplicate sample ID: {sample_id}")
        by_id[sample_id] = record
    missing = sorted(set(by_id) - set(schedule))
    unknown = sorted(set(schedule) - set(by_id))
    if missing or unknown or len(schedule) != len(records):
        raise ValueError(f"Train schedule must be an exact permutation; missing={missing}, unknown={unknown}.")
    return [by_id[sample_id] for sample_id in schedule], schedule


def unwrap_training_model(model):
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def compute_text_loss(
    logits: torch.Tensor,
    targets: torch.LongTensor,
    *,
    audio_end_token_id: int,
    audio_assistant_slot_token_id: int,
    eos_loss_weight: float = 1.0,
    eos_loss_mode: str = "token_weight",
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute text-channel CE while keeping continuation and stop observable.

    A packed TTS target contains one assistant-slot text target per acoustic
    frame followed by exactly one audio-end target.  A plain mean therefore
    gives the stopping decision only 1/(frames + 1) of the text objective.
    Weighting is deliberately based only on the target token ID; audio/VQ
    logits and losses never enter this helper.
    """
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise ValueError("Text logits/targets must have shapes (batch, seq, vocab)/(batch, seq).")
    if eos_loss_mode not in EOS_LOSS_MODES:
        raise ValueError(f"Unknown eos_loss_mode: {eos_loss_mode!r}.")
    valid = targets.ne(-100)
    if not valid.any():
        raise ValueError("Text loss requires at least one non-ignored target.")
    if not math.isfinite(eos_loss_weight) or eos_loss_weight <= 0:
        raise ValueError("`eos_loss_weight` must be finite and > 0.")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    per_token = F.cross_entropy(
        flat_logits.float(), flat_targets, ignore_index=-100, reduction="none"
    ).reshape_as(targets)
    stop = valid & targets.eq(int(audio_end_token_id))
    continuation = valid & targets.eq(int(audio_assistant_slot_token_id))
    unexpected = valid & ~stop & ~continuation
    if unexpected.any():
        unexpected_ids = sorted(set(targets[unexpected].detach().cpu().tolist()))
        raise ValueError(f"Text targets contain tokens other than slot/end: {unexpected_ids}.")
    stop_counts = stop.sum(dim=1)
    if not stop_counts.eq(1).all():
        raise ValueError(f"Each sample must contain exactly one audio_end target; got {stop_counts.tolist()}.")
    continuation_counts = continuation.sum(dim=1)
    if not continuation_counts.gt(0).all():
        raise ValueError(f"Each sample must contain at least one continuation target; got {continuation_counts.tolist()}.")

    if eos_loss_mode == "token_weight":
        weights = valid.to(dtype=per_token.dtype)
        weights = torch.where(stop, weights * float(eos_loss_weight), weights)
        loss = (per_token * weights).sum() / weights.sum()
    else:
        continue_per_sample = (per_token * continuation).sum(dim=1) / continuation_counts
        stop_per_sample = (per_token * stop).sum(dim=1)  # exactly one by contract
        loss = ((continue_per_sample + float(eos_loss_weight) * stop_per_sample)
                / (1.0 + float(eos_loss_weight))).mean()
    # Runtime termination is a binary decision between continuing with the
    # assistant slot and stopping with audio_end. Report that exact decision.
    slot_logits = logits[:, :, int(audio_assistant_slot_token_id)].float()
    stop_logits = logits[:, :, int(audio_end_token_id)].float()

    nan = torch.full((), float("nan"), device=logits.device, dtype=torch.float32)
    breakdown: Dict[str, torch.Tensor] = {}
    for name, mask in (("text_continue", continuation), ("text_stop", stop)):
        breakdown[f"{name}_loss"] = per_token[mask].mean().detach().float() if mask.any() else nan
        correct = slot_logits.gt(stop_logits) if name == "text_continue" else stop_logits.gt(slot_logits)
        signed_margin = slot_logits - stop_logits if name == "text_continue" else stop_logits - slot_logits
        breakdown[f"{name}_accuracy"] = correct[mask].float().mean().detach() if mask.any() else nan
        breakdown[f"{name}_margin"] = signed_margin[mask].mean().detach() if mask.any() else nan
        breakdown[f"{name}_count"] = mask.sum().detach().float()
    breakdown["text_sequence_count"] = torch.tensor(
        targets.shape[0], device=logits.device, dtype=torch.float32
    )
    return loss, breakdown


def compute_tail_weighted_channel_loss(
    channel_logits: torch.Tensor,
    channel_targets: torch.LongTensor,
    *,
    batch_size: int,
    seq_len: int,
    acoustic_frame_indices: torch.LongTensor,
    acoustic_tail_start_frames: torch.LongTensor,
    acoustic_tail_weighted_mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    """Return a sample-balanced body:tail 1:2 CE for one VQ head."""
    if tuple(acoustic_frame_indices.shape) != (batch_size, seq_len):
        raise ValueError("Acoustic frame indices must match the shifted label shape.")
    if tuple(acoustic_tail_start_frames.shape) != (batch_size,):
        raise ValueError("One acoustic tail boundary is required per sample.")
    per_token = F.cross_entropy(
        channel_logits.float(), channel_targets, ignore_index=-100, reduction="none"
    ).reshape(batch_size, seq_len)
    valid = channel_targets.reshape(batch_size, seq_len).ne(-100)
    frame_indices = acoustic_frame_indices.to(device=per_token.device)
    boundaries = acoustic_tail_start_frames.to(device=per_token.device)
    weighted_mask = (torch.ones(batch_size, dtype=torch.bool, device=per_token.device)
                     if acoustic_tail_weighted_mask is None
                     else acoustic_tail_weighted_mask.to(device=per_token.device))
    if tuple(weighted_mask.shape) != (batch_size,):
        raise ValueError("One acoustic tail mode is required per sample.")
    tail = weighted_mask[:, None] & (frame_indices >= boundaries[:, None])
    weights = torch.where(tail, 2.0, 1.0) * valid
    denominators = weights.sum(dim=1)
    if (denominators <= 0).any():
        raise ValueError("Every tail-weighted sample must contain acoustic targets.")
    return ((per_token * weights).sum(dim=1) / denominators).mean()


def compute_supervised_loss(
    model,
    *,
    input_ids: torch.LongTensor,
    attention_mask: torch.BoolTensor,
    labels: torch.LongTensor,
    channelwise_loss_weight: List[float],
    eos_loss_weight: float = 1.0,
    eos_loss_mode: str = "token_weight",
    acoustic_frame_indices: Optional[torch.LongTensor] = None,
    acoustic_tail_start_frames: Optional[torch.LongTensor] = None,
    acoustic_tail_weighted_mask: Optional[torch.BoolTensor] = None,
    return_breakdown: bool = False,
    return_global_hidden: bool = False,
    speaker_conditioner: Optional["SpeakerConditioner"] = None,
    speaker_embedding: Optional[torch.Tensor] = None,
):
    if speaker_conditioner is not None:
        if speaker_embedding is None:
            raise ValueError("speaker_conditioning requires a batch speaker embedding")
        speaker_conditioner.begin_batch(speaker_embedding)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    if speaker_conditioner is not None:
        speaker_conditioner.end_batch()
    global_hidden_states = outputs.global_hidden_states
    if global_hidden_states is None:
        raise RuntimeError("Model forward did not return global_hidden_states.")

    base_model = unwrap_training_model(model)
    batch_size, seq_len, hidden_size = global_hidden_states.shape
    n_vq = int(base_model.config.n_vq)
    flat_hidden = global_hidden_states.reshape(batch_size * seq_len, hidden_size)
    local_dtype = base_model.local_transformer.ln_f.weight.dtype
    flat_hidden = flat_hidden.to(dtype=local_dtype)

    flat_labels = labels.reshape(batch_size * seq_len, n_vq + 1)
    local_inputs = torch.zeros(
        (batch_size * seq_len, n_vq + 1, hidden_size),
        dtype=local_dtype,
        device=flat_hidden.device,
    )
    local_inputs[:, 0, :] = flat_hidden

    text_targets = flat_labels[:, 0]
    safe_text_targets = text_targets.masked_fill(text_targets.lt(0), int(base_model.config.pad_token_id))
    local_inputs[:, 1, :] = base_model.transformer.wte(safe_text_targets)

    audio_targets = flat_labels[:, 1:]
    for channel_index in range(n_vq - 1):
        teacher_ids = audio_targets[:, channel_index]
        valid_mask = (teacher_ids >= 0) & (teacher_ids < base_model.audio_embeddings[channel_index].num_embeddings)
        safe_ids = teacher_ids.masked_fill(~valid_mask, 0)
        channel_embeds = base_model.audio_embeddings[channel_index](safe_ids)
        channel_embeds = channel_embeds * valid_mask.unsqueeze(-1)
        local_inputs[:, channel_index + 2, :] = channel_embeds.to(dtype=local_dtype)

    local_attention_mask = torch.ones(
        (batch_size * seq_len, n_vq + 1),
        dtype=torch.bool,
        device=flat_hidden.device,
    )
    local_outputs = base_model.local_transformer(
        input_ids=None,
        attention_mask=local_attention_mask,
        position_ids=None,
        inputs_embeds=local_inputs,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cu_seqlens=None,
        num_sequences=None,
    )
    local_hidden_states = local_outputs.last_hidden_state

    total_loss = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
    total_weight = 0.0
    # Fixed keys are required for distributed diagnostics: separate ranks may
    # see batches where a padded channel has no valid targets, but every rank
    # must execute the same number and order of gather collectives.
    channel_losses: Dict[str, torch.Tensor] = {
        name: torch.full((), float("nan"), device=flat_hidden.device, dtype=torch.float32)
        for name in (
            "text",
            "text_continue_loss",
            "text_stop_loss",
            "text_continue_accuracy",
            "text_stop_accuracy",
            "text_continue_margin",
            "text_stop_margin",
            "text_continue_count",
            "text_stop_count",
            "text_sequence_count",
            *(f"vq{index}" for index in range(n_vq)),
        )
    }

    text_logits = base_model.text_lm_head(local_hidden_states[:, 0, :])
    if (text_targets != -100).any():
        text_loss, text_breakdown = compute_text_loss(
            text_logits.reshape(batch_size, seq_len, -1),
            labels[:, :, 0],
            audio_end_token_id=int(base_model.config.audio_end_token_id),
            audio_assistant_slot_token_id=int(base_model.config.audio_assistant_slot_token_id),
            eos_loss_weight=eos_loss_weight,
            eos_loss_mode=eos_loss_mode,
        )
        channel_losses["text"] = text_loss.detach().float()
        channel_losses.update(text_breakdown)
        total_loss = total_loss + float(channelwise_loss_weight[0]) * text_loss.float()
        total_weight += float(channelwise_loss_weight[0])

    for channel_index in range(n_vq):
        channel_targets = audio_targets[:, channel_index]
        if not (channel_targets != -100).any():
            continue
        channel_logits = base_model.audio_lm_heads[channel_index](local_hidden_states[:, channel_index + 1, :])
        if acoustic_tail_start_frames is None:
            # Preserve the legacy objective bit-for-bit when tail weighting is absent.
            channel_loss = F.cross_entropy(channel_logits.float(), channel_targets, ignore_index=-100)
        else:
            if acoustic_frame_indices is None:
                raise ValueError("Tail weighting requires acoustic frame indices.")
            channel_loss = compute_tail_weighted_channel_loss(
                channel_logits, channel_targets, batch_size=batch_size, seq_len=seq_len,
                acoustic_frame_indices=acoustic_frame_indices,
                acoustic_tail_start_frames=acoustic_tail_start_frames,
                acoustic_tail_weighted_mask=acoustic_tail_weighted_mask,
            )
        channel_losses[f"vq{channel_index}"] = channel_loss.detach().float()
        total_loss = total_loss + float(channelwise_loss_weight[channel_index + 1]) * channel_loss.float()
        total_weight += float(channelwise_loss_weight[channel_index + 1])

    if total_weight <= 0:
        raise RuntimeError("All labels are ignored; check dataset packing and max_length.")
    resolved_loss = total_loss / total_weight
    if return_breakdown:
        if return_global_hidden:
            return resolved_loss, channel_losses, global_hidden_states
        return resolved_loss, channel_losses
    if return_global_hidden:
        return resolved_loss, global_hidden_states
    return resolved_loss


def module_gradient_norms(model, *, loss_scale: float = 1.0) -> Dict[str, float]:
    """Report true pre-clip L2 norms without mutating loss-scaled gradients."""
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError(f"loss_scale must be finite and positive, got {loss_scale!r}")
    groups = {
        "local_transformer": "local_transformer.",
        # Output heads are weight-tied. named_parameters() reports each shared
        # tensor once under the embedding-side name, so group the two roles.
        "text_embedding_head": "transformer.wte.",
        "audio_embedding_heads": "audio_embeddings.",
        "global_transformer": "transformer.",
    }
    squared = {name: 0.0 for name in groups}
    squared["other"] = 0.0
    base_model = unwrap_training_model(model)
    for parameter_name, parameter in base_model.named_parameters():
        if parameter.grad is None:
            continue
        # PEFT prefixes names with e.g. base_model.model.; substring matching
        # keeps the same grouping for full-parameter and adapter training.
        group = next((name for name, marker in groups.items() if marker in parameter_name), "other")
        # GradScaler leaves FP16 gradients scaled until clipping/step. Divide
        # only the diagnostic value so Accelerate remains responsible for the
        # single in-place unscale operation in its normal optimizer path.
        norm = parameter.grad.detach().float().norm(2).item() / loss_scale
        squared[group] += norm * norm
    return {name: math.sqrt(value) for name, value in squared.items()}


def append_diagnostics(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_asset(model_path: str, filename: str) -> Optional[Path]:
    model_path_obj = Path(model_path)
    if model_path_obj.is_dir():
        candidate = model_path_obj / filename
        return candidate if candidate.exists() else None

    try:
        resolved = cached_file(
            model_path,
            filename,
            _raise_exceptions_for_missing_entries=False,
        )
    except OSError:
        return None

    if resolved is None:
        return None
    return Path(resolved)


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    tokenizer,
    model_path: str,
    codec_path: str,
    output_dir: Path,
    train_args: Dict[str, Any],
    global_step: int,
    epoch: int,
    content_head=None,
    speaker_conditioner=None,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = unwrap_training_model(model)
    if hasattr(unwrapped_model, "peft_config"):
        unwrapped_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        metadata = dict(train_args)
        metadata["saved_global_step"] = int(global_step)
        metadata["saved_epoch"] = int(epoch)
        metadata["saved_at"] = format_timestamp()
        metadata["checkpoint_dir"] = str(output_dir)
        metadata["checkpoint_type"] = "lora_adapter"
        if content_head is not None:
            head_path = output_dir / "content_head.pt"
            torch.save(
                {key: value.detach().cpu() for key, value in content_head.state_dict().items()},
                head_path,
            )
            metadata["content_head_file"] = head_path.name
            metadata["content_head_sha256"] = hashlib.sha256(head_path.read_bytes()).hexdigest()
        if speaker_conditioner is not None:
            conditioner_path = output_dir / "speaker_conditioner.pt"
            torch.save(
                {key: value.detach().cpu() for key, value in speaker_conditioner.state_dict().items()},
                conditioner_path,
            )
            metadata["speaker_conditioner_file"] = conditioner_path.name
            metadata["speaker_conditioner_sha256"] = hashlib.sha256(conditioner_path.read_bytes()).hexdigest()
        with open(output_dir / "finetune_config.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        return

    unwrapped_model.config.audio_tokenizer_pretrained_name_or_path = str(Path(codec_path).expanduser().resolve())
    unwrapped_model.config.save_pretrained(output_dir)
    state_dict = {
        key: value.detach().cpu()
        for key, value in unwrapped_model.state_dict().items()
    }
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)

    for filename in MODEL_SUPPORT_FILES:
        src = resolve_asset(model_path, filename)
        if src is not None and src.exists():
            shutil.copy2(src, output_dir / filename)

    metadata = dict(train_args)
    metadata["saved_global_step"] = int(global_step)
    metadata["saved_epoch"] = int(epoch)
    metadata["saved_at"] = format_timestamp()
    metadata["checkpoint_dir"] = str(output_dir)
    if content_head is not None:
        head_path = output_dir / "content_head.pt"
        torch.save(
            {key: value.detach().cpu() for key, value in content_head.state_dict().items()},
            head_path,
        )
        metadata["content_head_file"] = head_path.name
        metadata["content_head_sha256"] = hashlib.sha256(head_path.read_bytes()).hexdigest()
    if speaker_conditioner is not None:
        conditioner_path = output_dir / "speaker_conditioner.pt"
        torch.save(
            {key: value.detach().cpu() for key, value in speaker_conditioner.state_dict().items()},
            conditioner_path,
        )
        metadata["speaker_conditioner_file"] = conditioner_path.name
        metadata["speaker_conditioner_sha256"] = hashlib.sha256(conditioner_path.read_bytes()).hexdigest()
    with open(output_dir / "finetune_config.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_torch_backends()
    set_seed(args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=resolve_accelerate_mixed_precision(args.mixed_precision),
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.device.type != "cuda" and not args.allow_cpu:
        raise EnvironmentError(
            f"MOSS-TTS-Nano finetuning requires CUDA unless --allow-cpu is explicit; "
            f"Accelerate resolved device={accelerator.device}."
        )
    if args.pcgrad and accelerator.num_processes != 1:
        raise EnvironmentError("PCGrad is currently validated only for one CPU process or one GPU.")
    if args.train_schedule_json and accelerator.num_processes != 1:
        raise EnvironmentError("Scheduled training requires exactly one process for deterministic sample order.")

    model_dtype = resolve_torch_dtype(args.mixed_precision)
    attn_implementation = resolve_attn_implementation(args.attn_implementation, model_dtype)
    records_paths, records = load_jsonl_spec(args.train_jsonl)
    records, train_schedule = apply_train_schedule(records, args.train_schedule_json)
    has_tail_contract = any(record.get("acoustic_tail_mode") is not None
                            or record.get("acoustic_tail_start_frame") is not None for record in records)
    if has_tail_contract and not args.joint_formula_tail_weighting:
        raise ValueError("Acoustic tail fields require explicit --joint-formula-tail-weighting.")
    if args.joint_formula_tail_weighting and not args.joint_formula_pilot:
        raise ValueError("--joint-formula-tail-weighting requires --joint-formula-pilot.")
    validate_calibration_protection_enabled(records, pcgrad=args.pcgrad)
    protect_records_paths, protect_records = (load_jsonl_spec(args.protect_jsonl) if args.pcgrad else ([], []))
    behavior_records_paths, behavior_records = (
        load_jsonl_spec(args.behavior_protect_jsonl) if args.behavior_protect_jsonl else ([], [])
    )
    same_speaker_heldout_paths, same_speaker_heldout_records = (
        load_jsonl_spec(args.same_speaker_heldout_jsonl) if args.same_speaker_pilot else ([], [])
    )
    if train_schedule and (
        args.max_train_steps < len(train_schedule)
        or args.max_train_steps % len(train_schedule) != 0
    ):
        raise ValueError(
            "Scheduled training requires max_train_steps to be a positive multiple of the "
            f"schedule length ({len(train_schedule)}), got {args.max_train_steps}."
        )

    validate_round2_model_baseline(
        records, args.model_path,
        expected_sha256=args.calibration_source_model_sha256,
        lora_rank=args.lora_rank,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation(attn_implementation)
    if args.lora_rank > 0:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("LoRA requires `peft`; install it before using --lora-rank.") from exc
        modules_to_save = [item.strip() for item in args.lora_modules_to_save.split(",") if item.strip()]
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                modules_to_save=modules_to_save or None,
                bias="none",
            ),
        )

    content_alignment = None
    content_head = None
    if args.content_loss_weight > 0:
        content_alignment = load_content_alignment(args.content_alignment_jsonl)
        validate_alignment_covers_records(content_alignment, records)
        content_head = ContentHead(
            hidden_size=int(model.config.hidden_size),
            vocab=alignment_vocab(content_alignment),
        )

    speaker_embeddings = None
    speaker_conditioner = None
    speaker_reference_by_sample = {}
    if args.speaker_conditioning:
        speaker_embeddings = load_speaker_embeddings(args.speaker_embeddings_jsonl)
        base_for_conditioner = model
        while hasattr(base_for_conditioner, "base_model") and not hasattr(base_for_conditioner, "transformer"):
            base_for_conditioner = base_for_conditioner.base_model
        n_layers = len(base_for_conditioner.transformer.h)
        speaker_conditioner = SpeakerConditioner(
            embedding_dim=embedding_dim(speaker_embeddings),
            hidden_size=int(model.config.hidden_size),
            n_layers=n_layers,
            film_rank=32,
        )
        attach_speaker_conditioner(model, speaker_conditioner)
        for row in list(records) + list(protect_records):
            provenance = row.get("reference_provenance") or {}
            reference_id = provenance.get("id")
            if reference_id:
                speaker_reference_by_sample[str(row.get("id", ""))] = str(reference_id)

    dataset = MossTTSNanoSFTDataset(
        records,
        tokenizer=tokenizer,
        model_config=model.config,
        max_length=args.max_length,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=not bool(train_schedule),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )
    protect_dataloader = None
    behavior_protect_dataloader = None
    if args.pcgrad:
        protect_dataset = MossTTSNanoSFTDataset(
            protect_records,
            tokenizer=tokenizer,
            model_config=model.config,
            max_length=args.max_length,
        )
        protect_dataloader = DataLoader(
            protect_dataset,
            batch_size=args.per_device_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=protect_dataset.collate_fn,
        )
    if behavior_records:
        behavior_dataset = MossTTSNanoSFTDataset(
            behavior_records, tokenizer=tokenizer, model_config=model.config, max_length=args.max_length,
        )
        behavior_protect_dataloader = DataLoader(
            behavior_dataset, batch_size=args.per_device_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            collate_fn=behavior_dataset.collate_fn,
        )

    extra_train_modules = [module for module in (content_head, speaker_conditioner) if module is not None]
    optimizer = build_optimizer(model, args, extra_modules=extra_train_modules)
    global_batch_size = (
        args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    micro_batches_per_epoch = math.ceil(len(dataset) / (args.per_device_batch_size * accelerator.num_processes))
    optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or (args.num_epochs * optimizer_steps_per_epoch)
    warmup_steps = resolve_warmup_steps(args, max_train_steps)
    channelwise_loss_weight, protect_channelwise_loss_weight = resolve_objective_loss_weights(
        args,
        int(model.config.n_vq) + 1,
    )
    behavior_protect_channelwise_loss_weight = parse_channelwise_loss_weight(
        args.behavior_protect_channelwise_loss_weight, int(model.config.n_vq) + 1,
    )
    validate_calibration_objective(
        records, eos_loss_mode=args.eos_loss_mode,
        channelwise_loss_weight=channelwise_loss_weight,
    )
    validate_joint_formula_pilot(
        records, protect_records, enabled=args.joint_formula_pilot, pcgrad=args.pcgrad,
        protection_scope=args.calibration_protection_scope, behavior_rows=behavior_records,
        eos_loss_mode=args.eos_loss_mode, channel_weights=channelwise_loss_weight,
        acoustic_weights=protect_channelwise_loss_weight, schedule=train_schedule,
        max_train_steps=args.max_train_steps, lora_rank=args.lora_rank,
        model_path=args.model_path, lora_target_modules=args.lora_target_modules,
        lora_modules_to_save=args.lora_modules_to_save,
        prior_report_path=args.joint_formula_prior_report,
        prior_report_sha256=args.joint_formula_prior_report_sha256,
        tail_weighting=args.joint_formula_tail_weighting,
        v2_fail_report_path=args.joint_formula_v2_fail_report,
        v2_fail_report_sha256=args.joint_formula_v2_fail_report_sha256,
        trace_audit_path=args.joint_formula_trace_audit,
        trace_audit_sha256=args.joint_formula_trace_audit_sha256,
        alignment_report_path=args.joint_formula_alignment_report,
        alignment_report_sha256=args.joint_formula_alignment_report_sha256,
    )
    resolved_same_speaker_contract = None
    if args.same_speaker_pilot:
        resolved_same_speaker_contract = validate_same_speaker_pilot(
            records, same_speaker_heldout_records, protect_records,
            preflight_manifest_path=args.same_speaker_preflight_manifest,
            preflight_manifest_sha256=args.same_speaker_preflight_manifest_sha256,
            max_train_steps=args.max_train_steps, eos_loss_mode=args.eos_loss_mode,
            channel_weights=channelwise_loss_weight, protect_weights=protect_channelwise_loss_weight,
            pcgrad=args.pcgrad, lora_rank=args.lora_rank, model_path=args.model_path,
            lora_target_modules=args.lora_target_modules, lora_modules_to_save=args.lora_modules_to_save,
            tail_weighting=args.joint_formula_tail_weighting,
        )
    if args.pcgrad:
        validate_behavior_protection(
            records, protect_records, behavior_records,
            acoustic_weights=protect_channelwise_loss_weight,
            behavior_weights=behavior_protect_channelwise_loss_weight,
            eos_loss_mode=args.eos_loss_mode, train_schedule=train_schedule,
            protection_scope=args.calibration_protection_scope,
            expected_source_sha256=args.calibration_source_model_sha256,
            expected_source_revision=args.calibration_source_revision,
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    if protect_dataloader is None:
        model, content_head, speaker_conditioner, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, content_head, speaker_conditioner, optimizer, train_dataloader, lr_scheduler,
        )
    elif behavior_protect_dataloader is None:
        model, content_head, speaker_conditioner, optimizer, train_dataloader, protect_dataloader, lr_scheduler = accelerator.prepare(
            model, content_head, speaker_conditioner, optimizer, train_dataloader, protect_dataloader, lr_scheduler,
        )
    else:
        model, content_head, speaker_conditioner, optimizer, train_dataloader, protect_dataloader, behavior_protect_dataloader, lr_scheduler = accelerator.prepare(
            model, content_head, speaker_conditioner, optimizer, train_dataloader, protect_dataloader, behavior_protect_dataloader, lr_scheduler,
        )

    output_root = Path(args.output_dir)
    diagnostics_path = Path(args.diagnostics_jsonl) if args.diagnostics_jsonl else None
    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)

    train_args_to_save = vars(args).copy()
    train_args_to_save["resolved_warmup_steps"] = warmup_steps
    train_args_to_save["resolved_channelwise_loss_weight"] = channelwise_loss_weight
    train_args_to_save["resolved_protect_channelwise_loss_weight"] = protect_channelwise_loss_weight
    train_args_to_save["resolved_behavior_protect_channelwise_loss_weight"] = behavior_protect_channelwise_loss_weight
    train_args_to_save["global_batch_size"] = global_batch_size
    train_args_to_save["records_paths"] = [str(path.resolve()) for path in records_paths]
    train_args_to_save["resolved_train_schedule"] = train_schedule
    train_args_to_save["protect_records_paths"] = [str(path.resolve()) for path in protect_records_paths]
    train_args_to_save["behavior_protect_records_paths"] = [str(path.resolve()) for path in behavior_records_paths]
    train_args_to_save["same_speaker_heldout_records_paths"] = [str(path.resolve()) for path in same_speaker_heldout_paths]
    train_args_to_save["resolved_same_speaker_contract"] = resolved_same_speaker_contract
    if speaker_embeddings is not None:
        train_args_to_save["resolved_speaker_conditioning"] = {
            "embedding_dim": embedding_dim(speaker_embeddings),
            "reference_ids": sorted(speaker_embeddings),
            "film_rank": 32,
        }
    if content_alignment is not None:
        train_args_to_save["resolved_content_alignment"] = {
            sample_id: {
                "frames": entry["frames"],
                "vocab_sha256": hashlib.sha256(
                    json.dumps(list(entry["vocab"]), ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
            }
            for sample_id, entry in sorted(content_alignment.items())
        }
    train_args_to_save["attn_implementation"] = attn_implementation
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    train_args_to_save["trainable_parameters"] = trainable_parameters
    train_args_to_save["total_parameters"] = total_parameters

    accelerator.print(
        f"[{format_timestamp()}] [sft] loaded_records={len(dataset)} "
        f"device={accelerator.device} "
        f"num_processes={accelerator.num_processes} "
        f"global_batch_size={global_batch_size} "
        f"micro_batches_per_epoch={micro_batches_per_epoch} "
        f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} "
        f"max_train_steps={max_train_steps} "
        f"warmup_steps={warmup_steps} "
        f"attn={attn_implementation} "
        f"model_dtype={model_dtype}"
        f" trainable_parameters={trainable_parameters}/{total_parameters} "
        f"pcgrad={args.pcgrad} protect_records={len(protect_records)}"
    )

    global_step = 0
    completed_epochs = 0
    last_log_time = time.perf_counter()
    last_logged_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                speaker_batch_embedding = None
                if speaker_conditioner is not None:
                    speaker_batch_embedding = resolve_batch_embedding(
                        batch["sample_ids"],
                        reference_id_by_sample=speaker_reference_by_sample,
                        embeddings=speaker_embeddings,
                    ).to(batch["input_ids"].device)
                content_enabled = content_head is not None
                if content_enabled:
                    loss, channel_losses, global_hidden = compute_supervised_loss(
                        model,
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        channelwise_loss_weight=channelwise_loss_weight,
                        eos_loss_weight=args.eos_loss_weight,
                        eos_loss_mode=args.eos_loss_mode,
                        acoustic_frame_indices=(batch.get("acoustic_frame_indices")
                                                if args.joint_formula_tail_weighting else None),
                        acoustic_tail_start_frames=(batch.get("acoustic_tail_start_frames")
                                                    if args.joint_formula_tail_weighting else None),
                        acoustic_tail_weighted_mask=(batch.get("acoustic_tail_weighted_mask")
                                                     if args.joint_formula_tail_weighting else None),
                        return_breakdown=True,
                        return_global_hidden=True,
                        speaker_conditioner=speaker_conditioner,
                        speaker_embedding=speaker_batch_embedding,
                    )
                else:
                    loss, channel_losses = compute_supervised_loss(
                        model,
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        channelwise_loss_weight=channelwise_loss_weight,
                        eos_loss_weight=args.eos_loss_weight,
                        eos_loss_mode=args.eos_loss_mode,
                        acoustic_frame_indices=(batch.get("acoustic_frame_indices")
                                                if args.joint_formula_tail_weighting else None),
                        acoustic_tail_start_frames=(batch.get("acoustic_tail_start_frames")
                                                    if args.joint_formula_tail_weighting else None),
                        acoustic_tail_weighted_mask=(batch.get("acoustic_tail_weighted_mask")
                                                     if args.joint_formula_tail_weighting else None),
                        return_breakdown=True,
                        speaker_conditioner=speaker_conditioner,
                        speaker_embedding=speaker_batch_embedding,
                    )
                    global_hidden = None
                content_metrics = None
                if content_enabled:
                    content_targets = build_content_targets(
                        batch["sample_ids"],
                        batch["acoustic_frame_indices"],
                        content_alignment,
                        device=global_hidden.device,
                    )
                    content_loss, content_metrics = compute_content_loss(
                        content_head, global_hidden, content_targets,
                    )
                    loss = loss + args.content_loss_weight * content_loss
                projection_report = None
                if protect_dataloader is None:
                    accelerator.backward(loss)
                else:
                    acoustic_batches = list(protect_dataloader)
                    behavior_batches = list(behavior_protect_dataloader or [])
                    protector_loss_factories = [
                        lambda item=item: compute_supervised_loss(
                            model, input_ids=item["input_ids"], attention_mask=item["attention_mask"],
                            labels=item["labels"], channelwise_loss_weight=protect_channelwise_loss_weight,
                            eos_loss_weight=args.eos_loss_weight, eos_loss_mode=args.eos_loss_mode,
                            speaker_conditioner=speaker_conditioner,
                            speaker_embedding=(resolve_batch_embedding(
                                item["sample_ids"], reference_id_by_sample=speaker_reference_by_sample,
                                embeddings=speaker_embeddings).to(item["input_ids"].device)
                                if speaker_conditioner is not None else None),
                        ) for item in acoustic_batches
                    ]
                    protector_loss_factories.extend(
                        lambda item=item: compute_supervised_loss(
                            model, input_ids=item["input_ids"], attention_mask=item["attention_mask"],
                            labels=item["labels"], channelwise_loss_weight=behavior_protect_channelwise_loss_weight,
                            eos_loss_weight=args.eos_loss_weight, eos_loss_mode="sequence_balanced",
                            speaker_conditioner=speaker_conditioner,
                            speaker_embedding=(resolve_batch_embedding(
                                item["sample_ids"], reference_id_by_sample=speaker_reference_by_sample,
                                embeddings=speaker_embeddings).to(item["input_ids"].device)
                                if speaker_conditioner is not None else None),
                        ) for item in behavior_batches
                    )
                    projection_report = pcgrad_backward(
                        accelerator=accelerator,
                        model=model,
                        teacher_loss=loss,
                        protector_loss_factories=protector_loss_factories,
                    )

                gradient_norms = None
                if accelerator.sync_gradients and diagnostics_path is not None:
                    scaler = getattr(accelerator, "scaler", None)
                    loss_scale = float(scaler.get_scale()) if scaler is not None else 1.0
                    gradient_norms = module_gradient_norms(model, loss_scale=loss_scale)

                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    clip_parameters = list(model.parameters())
                    for extra_module in extra_train_modules:
                        clip_parameters.extend(extra_module.parameters())
                    accelerator.clip_grad_norm_(clip_parameters, args.max_grad_norm)

                if accelerator.sync_gradients:
                    optimizer.step()
                    if not getattr(optimizer, "step_was_skipped", False):
                        lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0:
                    now = time.perf_counter()
                    steps_since_last_log = max(global_step - last_logged_step, 1)
                    elapsed = max(now - last_log_time, 1e-12)
                    last_log_time = now
                    last_logged_step = global_step
                    step_time = elapsed / steps_since_last_log
                    steps_per_sec = steps_since_last_log / elapsed
                    samples_per_sec = (global_batch_size * steps_since_last_log) / elapsed
                    eta_seconds = max(max_train_steps - global_step, 0) / steps_per_sec
                    logged_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    lr_val = lr_scheduler.get_last_lr()[0]
                    accelerator.print(
                        f"[{format_timestamp()}] "
                        f"epoch={epoch} step={global_step}/{max_train_steps} "
                        f"loss={logged_loss:.4f} "
                        f"lr={lr_val:.2e} "
                        f"step_time={step_time:.2f}s "
                        f"steps_per_sec={steps_per_sec:.3f} "
                        f"samples_per_sec={samples_per_sec:.2f} "
                        f"eta={format_duration(eta_seconds)}"
                    )
                    if diagnostics_path is not None:
                        gathered_channels = {}
                        for name in channel_losses:
                            gathered = accelerator.gather(channel_losses[name].reshape(1))
                            finite = gathered[torch.isfinite(gathered)]
                            gathered_channels[name] = finite.mean().item() if finite.numel() else None
                        if accelerator.is_main_process:
                            append_diagnostics(diagnostics_path, {
                                "epoch": epoch,
                                "step": global_step,
                                "loss": logged_loss,
                                "learning_rate": lr_val,
                                "channel_loss": gathered_channels,
                                "content": content_metrics,
                                "gradient_norm_pre_clip": gradient_norms or {},
                                "teacher_sample_ids": list(batch.get("sample_ids", [])) if args.pcgrad else None,
                                "protector_sample_ids": [sid for item in acoustic_batches for sid in item.get("sample_ids", [])] if args.pcgrad else None,
                                "acoustic_protector_sample_ids": [sid for item in acoustic_batches for sid in item.get("sample_ids", [])] if args.pcgrad else None,
                                "behavior_protector_sample_ids": [sid for item in behavior_batches for sid in item.get("sample_ids", [])] if args.pcgrad else None,
                                "pcgrad": None if projection_report is None else {
                                    "iterations": projection_report.iterations,
                                    "original_minimum_dot": projection_report.original_minimum_dot,
                                    "minimum_dot": projection_report.minimum_dot,
                                    "retained_norm_ratio": projection_report.retained_norm_ratio,
                                    "feasibility_tolerance": projection_report.feasibility_tolerance,
                                    "constraint_violation_count": sum(
                                        not is_feasible_dot(dot, tolerance=projection_report.feasibility_tolerance)
                                        for dot in projection_report.dots
                                    ),
                                    "original_constraint_dots": list(projection_report.original_dots),
                                    "constraint_dots": list(projection_report.dots),
                                    "acoustic_original_dots": list(projection_report.original_dots[:len(acoustic_batches)]),
                                    "acoustic_constraint_dots": list(projection_report.dots[:len(acoustic_batches)]),
                                    "behavior_original_dots": list(projection_report.original_dots[len(acoustic_batches):]),
                                    "behavior_constraint_dots": list(projection_report.dots[len(acoustic_batches):]),
                                    "constraint_kinds": (
                                        ["acoustic"] * len(acoustic_batches) + ["behavior"] * len(behavior_batches)
                                    ),
                                },
                            })

                if global_step >= max_train_steps:
                    break

        if (epoch + 1) % args.save_every_epochs == 0 or global_step >= max_train_steps:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                model_path=args.model_path,
                codec_path=args.codec_path,
                output_dir=output_root / f"checkpoint-epoch-{epoch + 1}",
                train_args=train_args_to_save,
                global_step=global_step,
                epoch=epoch + 1,
                content_head=content_head,
                speaker_conditioner=speaker_conditioner,
            )
        completed_epochs = epoch + 1

        if global_step >= max_train_steps:
            break

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
        codec_path=args.codec_path,
        output_dir=output_root / "checkpoint-last",
        train_args=train_args_to_save,
        global_step=global_step,
        epoch=completed_epochs,
        content_head=content_head,
        speaker_conditioner=speaker_conditioner,
    )
    accelerator.print(
        f"[{format_timestamp()}] [sft] finished "
        f"global_step={global_step} saved_epochs={completed_epochs} output_dir={output_root}"
    )


if __name__ == "__main__":
    main()
