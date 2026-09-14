# Joint v3 failure audit (CPU, read-only)

## Scope and artifacts

This audit did not train a model, use a GPU, promote a checkpoint, or play audio.
It inspected the v2/v3 merged checkpoints, the five authorized Doubao hydrate
targets, eight Junhao acoustic protect rows, and the v3 exact-Cu free-run trace.
The v3 checkpoint under test has SHA-256
`5cc3bc55a59403030436427dabcd176ee83e58ce6122a84a4b58277dafb5efc2`.
Target ranks below use raw head logits before repetition-penalty sampling.
Disposable CPU trace SHA-256 values are: teacher-forced
`5c08d172f6b6a17eebdfbd4c26f13628b147f9ab10ed578afea907783501fab6`,
free-hidden `cd829c91ee8b2ae5476060e3ec6277b7b5c8613f21d03b209b6cc6a039654aa4`,
exact-text/no-reference
`c539530240a3186869c34042c1e8c7454a150bf969363f9a2a0d657a37ec7df2`,
and exact-text/Junhao-reference
`f0f7ee02930d6d1b310009b0b999df3c0c7085f56c45a574b496b4276a55692a`.

## The apparent frame-24 fork is not the first teacher divergence

The saved v2 and v3 greedy code trajectories first differ from each other at
frame 24. Both, however, differ from the 68-frame Doubao Cu target at frame 0.
Across 68 frames x 16 VQ heads, each free run puts only 1/1,088 teacher targets
at rank 1. At frame 0 all 16 heads miss; their target ranks are identically
`[103,232,331,970,124,127,818,614,171,432,719,608,442,506,73,640]`.

| Trace | target top-1 | mean / median rank (of 1024) | mean target-vs-max margin |
|---|---:|---:|---:|
| v2 free | 1/1,088 | 476.25 / 466.5 | -8.81 |
| v3 free | 1/1,088 | 483.85 / 474 | -10.00 |
| v3 teacher-forced, same Junhao prompt | 53/1,088 | 113.36 / 45.5 | -1.94 |

Teacher forcing improves the target rank substantially, proving history-induced
error accumulation, but 53/1,088 (4.9%) is still not target acquisition.
Consequently exposure bias is an amplifier, not a sufficient root cause.

With the same v3 model and Junhao prompt, teacher-forced versus free-run global
last-layer hidden-state cosine is 1.000 at frame 0, averages 0.398 at frames
1-23, 0.256 at frames 24-51, and 0.165 at frames 52-67. The corresponding
16-head logit cosine averages 0.263 at frames 1-23, 0.088 at frames 24-51, and
-0.100 in the weighted tail (52-67). This is the expected autoregressive drift,
but it starts after a first decision whose teacher code is already low-ranked.

## Conditioning and speaker conflict

All five Doubao training rows and all eight Junhao protect rows omit
`ref_audio_codes`; the packed training prompt therefore says Reference `None`.
Product evaluation instead supplies a Junhao prompt wav in `voice_clone` mode.
The same global LoRA is thus optimized on unreferenced Doubao codec labels,
constrained by unreferenced Junhao codec labels, and evaluated under a third
condition containing Junhao reference codes.

An exact-spoken-text free-run A/B isolates that condition:

| Condition | target top-1 | mean / median rank | margin | unique frames | repeat onset | SenseVoice no-ITN |
|---|---:|---:|---:|---:|---:|---|
| training-like Reference None | 0/1,088 | 500.55 / 494 | -16.05 | 13/68 | 12 | `苏苏` |
| Junhao reference | 2/1,088 | 486.88 / 475 | -13.01 | 20/68 | 18 | `扩收四` |

The two trajectories diverge at frame 0 and share only 162/1,104 generated
tokens. Removing the reference does not repair the model; it makes collapse
earlier. The condition mismatch is therefore real but is not a standalone fix.

As a distribution diagnostic, the five Doubao targets contain 351 frames and
the Junhao protect set 239. Per-codebook smoothed Jensen-Shannon divergence is
0.7057-0.8118 bits (mean 0.7625); only 21.2% of the teacher's observed code
vocabulary overlaps the protector vocabulary on average. This is consistent
with a cross-speaker/content target conflict, but sparse codec histograms also
encode phonetic content, so it is supporting evidence rather than a speaker
identity metric. CAMPPlus, WeSpeaker, FunASR, and ModelScope speaker encoders
are not installed in the isolated Contabo environment; no speaker-distance
number is fabricated.

## Causal judgment and stop rules

The dominant failure is off-manifold target supervision under mismatched prompt
and speaker conditions: even teacher-forced target codes are weak, and free-run
is wrong at its first codec decision. Exposure bias then magnifies that error
until hidden states and logits become nearly orthogonal in the tail. Historical
EOS-only weighting, three DAgger rounds, an external stop controller, increasing
total acoustic weight from 0.125 to 0.5, and v3 tail weighting all failed for the
same reason: they adjust duration or later frames without fixing frame-0 content
and conditioning.

Do not run Round 4 DAgger; raise EOS/acoustic/tail weights; add steps; truncate at
a guessed duration; train Doubao codes under a Junhao reference; open a GPU; or
connect v3 to the App. Those moves either repeat a failed axis or strengthen the
speaker/condition contradiction.

The single next experiment should be a CPU-only, optimizer-free 2x2 likelihood
probe using existing assets: for one Doubao hydrate target and one Junhao
hydrate target, compute all 16 VQ teacher-forced ranks under (a) Reference None
and (b) a short independent same-speaker reference from another authorized row.
Keep exact spoken text and all model weights fixed. Proceed to any SFT only if
the same-speaker condition improves frame-0 and full-sequence ranks for both
speakers without swapping the preferred target; otherwise the joint
cross-speaker codec objective is rejected and the data/voice strategy must be
redesigned.
