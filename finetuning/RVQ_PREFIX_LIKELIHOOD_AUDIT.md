# Frozen MOSS RVQ-prefix likelihood audit

## Contract and provenance

This was an optimizer-free CPU probe. It did not train, generate, decode, or
play audio and did not use a GPU. The frozen base weights SHA-256 is
`bff330205cea9136e5857c76417c294e0381a36ee13155bb5f96b48efad8da90`.
The full per-codebook artifact is retained on Contabo at
`/home/deploy/cue-u-training/moss-tts/audits/rvq-prefix-likelihood-v1/report.json`,
SHA-256 `b0d9ee3959df8e1fb64399e31d6ec63e929163cbcd1a7d469e1a22034932cf88`.

The near-parallel content pair is `CoCl₂·6H₂O`: Doubao target
`doubao_hydrate_hexahydrate_spelled_v2` (82 frames) and Junhao target
`junhao_hydrate` (51 frames). References are strictly independent utterances:
Doubao `doubao_hydrate_pentahydrate` and Junhao `junhao_caoh`. Target and
reference asset hashes differ; no target codes are used as their own reference.

For each target/reference combination, the probe keeps only the first
1/2/4/8/16 codebook embeddings in reference frames and prior target-history
frames before the global Transformer. It retains all 16 labels and the normal
within-frame teacher-forced local chain, so CE/rank tests how much predictive
information each legal RVQ prefix supplies; it does not claim to measure audio
quality. Hidden cosine is averaged over target frames. Exact spoken text and
weights remain fixed.

## Predeclared decision rule

A content prefix must satisfy all of the following for both target speakers:

1. preserve at least 95% of full-RVQ likelihood quality (CE no more than 5%
   above RVQ16 and mean rank no more than 10% above RVQ16);
2. preserve global state (cosine versus the same-reference RVQ16 >=0.95);
3. be reference-speaker invariant (same-target hidden cosine >=0.95 and CE
   difference between matched and cross-speaker references <=0.10);
4. achieve the conditions at a prefix shorter than 16; RVQ16 is not a separated
   content subspace.

## Aggregate results

Values are all-16-head teacher-forced averages. `match/cross CE gap` is
cross-reference CE minus matched-speaker-reference CE. Positive means the
independent same-speaker reference is preferred.

| Target | RVQ prefix | matched CE | matched rank | top1 / labels | hidden vs full | match/cross CE gap | cross-ref hidden cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Doubao | 1 | 7.280 | 392.8 | 10/1312 | -0.133 | +0.031 | 0.965 |
| Doubao | 2 | 7.475 | 404.3 | 10/1312 | -0.095 | +0.150 | 0.887 |
| Doubao | 4 | 7.928 | 410.6 | 5/1312 | 0.131 | -0.016 | 0.835 |
| Doubao | 8 | 6.072 | 187.6 | 36/1312 | 0.744 | -0.023 | 0.934 |
| Doubao | 16 | 5.175 | 95.2 | 87/1312 | 1.000 | +0.225 | 0.939 |
| Junhao | 1 | 7.449 | 411.6 | 3/816 | -0.108 | -0.011 | 0.956 |
| Junhao | 2 | 7.673 | 422.9 | 3/816 | -0.035 | -0.075 | 0.863 |
| Junhao | 4 | 7.332 | 359.5 | 8/816 | 0.180 | +0.077 | 0.843 |
| Junhao | 8 | 5.760 | 146.8 | 25/816 | 0.765 | +0.215 | 0.913 |
| Junhao | 16 | 4.872 | 74.3 | 72/816 | 1.000 | +0.324 | 0.938 |

Cross-reference likelihood at RVQ16 is also worse for both targets: Doubao CE
5.400/rank 116.8/top1 73 and Junhao CE 5.196/rank 99.5/top1 48. This symmetric
same-speaker preference is direct evidence that codec/reference conditioning
contains speaker-dependent information across the complete stack.

## Finding

No prefix passes. RVQ1/2/4 destroy both likelihood and global state. RVQ8 is the
only plausible compression point, but its CE is 17.3% above full for Doubao and
18.2% above full for Junhao; ranks are 97% and 98% worse, while hidden cosines
are only 0.744 and 0.765. It therefore cannot be called a preserved content
representation. RVQ16 restores likelihood but is not a subset, and its matched
speaker advantage plus cross-reference hidden cosine below 0.95 shows retained
speaker dependence.

The high cross-reference cosine at RVQ1 does not rescue it: the text and common
target history dominate the direction while target likelihood has collapsed to
near-random ranks. Hidden cosine must be interpreted jointly with CE/rank.

Accordingly, current MOSS RVQ codes do not expose an empirically defensible
content-only subspace that can be distilled while preserving Junhao timbre.
Do not train VQ0, VQ0-3, or VQ0-7 as "content layers"; do not freeze later heads
and claim voice preservation; and do not escalate this probe to GPU.

The minimal next experiment is no longer another codebook split. Introduce an
explicit content target outside the codec (authorized phoneme sequence or frozen
ASR/text hidden representation), keep a separate Junhao reference conditioner,
and first run an optimizer-free predictability probe: the content representation
must remain invariant across the two speakers while speaker identity remains
separable only in the conditioning branch. Only after that identifiability gate
passes is a small CPU distillation pilot justified.
