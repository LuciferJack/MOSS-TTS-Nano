# Joint formula content + acoustic SFT CPU pilot

EOS-only DAgger stopped after Round 3. This experiment starts from the frozen
accepted base, attaches a fresh formula-only LoRA, and jointly teaches the five
authorized hydrate utterances using their exact spoken text and MOSS codec codes.
It is not Round 4. Ordinary/acronym traffic always routes to the unmodified base.

## Single-variable objective

MOSS has one text head and 16 residual VQ heads. Start with
`--channelwise-loss-weight 1,0.125`: the trainer expands this to `1` for text and
`0.0078125` for each VQ layer. After loss normalization, text contributes 88.89%
and all acoustic layers together 11.11% (0.694% per VQ layer). All VQ layers stay
nonzero because later residual codebooks still carry timbre/detail; enabling only
VQ0 would change both objective size and spectral detail allocation. Keep
`sequence_balanced` EOS, EOS weight 1, learning rate and all decode settings fixed.

If and only if the complete `0.125` candidate report has verdict `content_fail`,
v2 may change the single variable to `--channelwise-loss-weight 1,0.5`. The
report path and SHA-256 are mandatory; the trainer rehashes and parses it through
`--joint-formula-prior-report` and `--joint-formula-prior-report-sha256`.
The report itself must bind full SHA-256 values for the v1 candidate weights and
its content-evaluation manifest.
At total `0.5`, each VQ weight is `0.03125`; normalized contributions are 66.67%
text, 33.33% acoustic total, and 2.083% per VQ. No intermediate or larger weight
is accepted, and every other setting remains identical to v1.

v3 does not raise the acoustic total again. It keeps `1,0.5` and changes only
the within-sample frame allocation. Every authorized row explicitly declares
`acoustic_tail_mode=aligned_weighted|legacy_unweighted`. An aligned row must
declare `acoustic_tail_start_frame` satisfying `0 < boundary < T`. For each of the 16
VQ heads the trainer computes unreduced CE, weights body frames 1 and tail
frames 2, then uses `sum(weight * CE) / sum(weight)` per sample before averaging
samples. Padding and the text EOS position are excluded. Without the explicit
tail switch the legacy CE path is unchanged bit-for-bit.

The v3 switch is fail-closed: `--joint-formula-tail-weighting` requires both the
hashed v2 (`0.5`) `content_fail` report and the hashed v1/v2 trace audit via
`--joint-formula-v2-fail-report{,-sha256}` and
`--joint-formula-trace-audit{,-sha256}`. It also retains v2's hashed v1 failure
arguments. This binds the change to observed late-content loss and prevents an
unreviewed increase to acoustic weight.

The independent teacher alignment report is also mandatory through
`--joint-formula-alignment-report{,-sha256}`. Each row records that report hash,
the two-method interval, `policy=conservative_lower_bound`, the selected lower
frame, and uncertainty in frames. `aligned_weighted` accepts width at most one,
including a report-level fail-closed verdict caused by unstable sub-frame ASR,
and conservatively selects the lower frame. `legacy_unweighted` requires width
greater than one, `reason=alignment_uncertain`, no selected frame and no boundary;
that row retains ordinary unweighted CE. Missing or invented modes are rejected.
The bound v2 alignment report SHA-256 is
`50436f3037b4d457307f41f470fa12106d02985d106651d3b8830348f77c234e`.
Its mixed plan is: dihydrate 55, decahydrate 44, and pentahydrate `[52,53] -> 52`
aligned-weighted; heptahydrate `[48,50]` and hexahydrate `[58,61]`
legacy-unweighted.

Use exactly five scheduled optimizer steps, batch 1, accumulation 1, then stop
and evaluate. Do not silently add epochs. Early reject immediately on non-finite
loss, PCGrad dot below `-1e-7`, retained teacher-gradient norm below 5%, any
formula content/EOS regression, or acoustic-protect regression. A candidate is
eligible only when all five formula outputs finish with complete final token and
Chinese semantic tail; no average can hide one failure.

## Voice-drift risk and gates

Only the 12-layer global AR attention/MLP LoRA is trainable; embeddings, local
transformer and text/VQ heads remain frozen. Nevertheless every VQ CE gradient
flows through global hidden states, so voice, rhythm and duration can drift. Eight
independent Junhao acoustic rows constrain the update with acoustic-only PCGrad
(`0,1`), preventing first-order loss increases but not guaranteeing perceptual
identity. Require all eight constraint dots within tolerance and retention >=5%,
speaker-embedding cosine >=0.98 versus base, median F0 shift <=5%, duration shift
<=10%, plus human no-tremor/noise approval. These are candidate gates, not claims
that the five-row pilot already generalizes.

Required switches are `--joint-formula-pilot --pcgrad
--calibration-protection-scope formula_scoped --protect-jsonl <8 rows>
--protect-channelwise-loss-weight 0,1 --channelwise-loss-weight 1,0.125
--eos-loss-mode sequence_balanced --eos-loss-weight 1 --lora-rank <positive>
--train-schedule-json <five exact IDs> --max-train-steps 5`. Do not pass behavior
protection rows: ordinary text is isolated structurally by base routing and must
still pass an external regression suite before accepting the adapter.
