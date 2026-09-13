# Exposure-bias repair for MOSS stop decisions

## Causal status

The seven-step sequence-balanced run covered all five hydrate teachers. Its
final decahydrate example had teacher-forced stop accuracy 1 and signed stop
margin `+2.5487`, while free-running hydrate inference still reached the
120-frame cap. The raw/spoken-normalized text mismatch was then isolated. With
exact teacher normalization held fixed, the free-running failure remained.
The residual failure is therefore a generated-prefix exposure mismatch, not
evidence that the acoustic teacher or EOS label is absent.

## Minimal frozen-policy EOS calibration

1. Freeze a source checkpoint and generate Junhao codec rows with the exact
   deployment prompt and decoding configuration.
2. Use an authorized external teacher only as the timing oracle that selects
   the target boundary. Never encode its waveform as a training target.
3. Retain the self-generated prefix through that boundary as
   `self_generated_prefix_codes`, not `audio_codes`.
4. Supervise assistant-slot on preceding text-channel rows and audio-end at the
   boundary, with sequence-balanced EOS loss and channel weights `1,0,...`.
5. Interleave pinned acronym/ordinary replay and reject candidates that regress
   those free-run boundaries or speech quality.

This is offline scheduled sampling with a frozen behavior policy. It exercises
the inference state trajectory without back-propagating through generation and
without cross-model acoustic-code distillation.

## Fail-closed row contract

Every calibration row declares role `self_generated_prefix_eos_calibration`,
source `self_generated_junhao_prefix`, voice `Junhao`, an immutable generator
revision and checkpoint SHA-256. The exact generation prompt and canonical JSON
generation config must match their recorded hashes; prompt must equal training
text and prompt-altering metadata is forbidden. Exact Junhao reference codes and
the generated prefix codes must also match recorded hashes. Reference identity,
asset hash and an independently recorded codec hash are mandatory; copying the
generated prefix (including the historical one-frame truncation) into the reference
field is rejected. Boundary provenance names
the provider, training authorization and teacher-asset SHA-256. The positive
`target_boundary_frame` must equal the retained prefix length, and
`codec_loss_eligible` must be false. `audio_codes` is forbidden.

The trainer rejects calibration rows unless EOS mode is `sequence_balanced`
and effective channel weights are exactly `1,0,...`. Collation independently
masks every VQ label on calibration rows. Generated codes therefore remain
conditioning context only, while ordinary/acronym replay rows retain normal VQ
labels. Missing/mismatched provenance, malformed codes, duplicate IDs and EOS
truncation all fail closed.

## Dual-protection feasibility

Dual PCGrad treats each acoustic and behavior protector as an independent
constraint. Gradients and dot products are evaluated in FP32. The formal
machine criterion is `dot >= -1e-7`, not a string-level `dot >= 0`; tiny
negative values inside this absolute tolerance are floating-point residuals.
Diagnostics record `feasibility_tolerance` and `constraint_violation_count`,
and the run fails if any dot is below the recorded tolerance or if less than
5% of the teacher-gradient norm remains.

## DAgger round 2 (formula-scoped adapter)

Round 2 must regenerate exactly five prefixes from the latest accepted merged
round-1 candidate, not from base and not from its pre-merge adapter. The current
candidate weight digest is
`9e7a3f5c4c86a3617b75cbdeaa1c40b92fd54b1f59761891646e00967613df38`;
the immutable source revision must also be supplied independently. Every row
records round `2`, that same generator SHA/revision, a matching
`parent_candidate_sha256`, and `adapter_initialization=fresh_on_merged_parent`.
The trainer hashes the actual `pytorch_model.bin`, rejects adapter directories,
and attaches a new LoRA. This prevents accidental adapter stacking or stale
policy trajectories.

Use one audited five-ID schedule, one row per optimizer step, exactly once. The
professional formula adapter retains all eight independent acoustic PCGrad
constraints (`0,1`) but excludes the two general behavior rows from its optimizer
geometry; acronym/general behavior remain external acceptance gates. After five
steps, merge the new adapter into the exact parent and evaluate the fixed suite.

Accept a round only if all five raw and normalized formula cases stop before the
cap, stop margins are positive, every acoustic constraint meets the recorded
PCGrad tolerance/retention gates, and acronym/general free-run plus audio quality
do not regress. Run at most one five-step update before regenerating trajectories
from the newly accepted merge. Reject on any regression, never reuse stale
prefixes, and cap iterative rounds at three. These rules prevent alternating
between stale-policy over-correction and a newly exposed continuation attractor.
