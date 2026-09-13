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
the generated prefix codes must also match recorded hashes. Boundary provenance names
the provider, training authorization and teacher-asset SHA-256. The positive
`target_boundary_frame` must equal the retained prefix length, and
`codec_loss_eligible` must be false. `audio_codes` is forbidden.

The trainer rejects calibration rows unless EOS mode is `sequence_balanced`
and effective channel weights are exactly `1,0,...`. Collation independently
masks every VQ label on calibration rows. Generated codes therefore remain
conditioning context only, while ordinary/acronym replay rows retain normal VQ
labels. Missing/mismatched provenance, malformed codes, duplicate IDs and EOS
truncation all fail closed.
