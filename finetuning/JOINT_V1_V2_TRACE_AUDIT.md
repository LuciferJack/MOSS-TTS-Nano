# Joint formula v1/v2 trace and weight audit

This is a read-only CPU audit. No checkpoint was modified and no inference was
rerun. The compared CuSO4 hydrate prompt, Junhao reference and greedy 120-frame
limit are identical. v1 merged weights are
`7bbc79275369bcac84a8c3ac4e38c2ffbec50f8c52355157b181071aae22acd0`;
v2 weights are
`949e3d78a8066180625f83d882b14b3d8fb140ff075efc103691c07f84969b6a`.
The v1 trace is bound by SHA-256
`ad2395d979c73973728e19623ba583485775bd02dd0df895426043117e24fa45`;
the v2 full-logit tensor is bound by
`2c36b7f588d8592f764af6db68f8de3cb071c7dd382b1a63f8db476734a51cdc`.

## Findings

- v1 and v2 generated audio codes are exactly equal at every one of 120 frames
  and all 16 VQ layers: 1,920/1,920 equality, with no first divergence.
- Both disagree with the 68-frame pentahydrate teacher from frame zero. Across
  the shared 68 frames only 1/1,088 codes matches (VQ15); no full frame matches.
  Acoustic supervision therefore did not copy the teacher content trajectory.
- The LoRA vectors are nearly identical: norms `5.666355` and `5.666364`, delta
  norm `0.004547`. Increasing the acoustic objective changed parameters, but not
  enough to cross a single greedy code decision on this trace.
- Text decisions are also effectively identical. Maximum v1/v2 absolute logit
  differences are `0.001061` for assistant-slot and `0.002428` for audio-end.
  Neither run selects native EOS. Both are closest at frame 103 with end-minus-slot
  about `-7.3218`; at frame 119 it remains about `-7.5144`.
- Teacher-forced diagnostics show only tiny v2 improvement: by steps 2..5 mean
  VQ CE deltas versus v1 are `-0.000101`, `-0.000115`, `-0.000169`, and
  `-0.000365`; maximum per-layer absolute delta is `0.001213`. Stop-margin deltas
  become slightly worse (`-0.00057` to `-0.00547`). Teacher forcing therefore
  does not predict a repaired free-running trajectory.
- Free run enters a multi-layer repetition attractor early. Identical full-code
  frames repeat at 32–37, 38–43, 51–62, 63–66, 67–82, 84–85, 86–91,
  92–93 and 94–119. The first six-frame collapse starts at frame 32, and the
  final code frame repeats for 26 frames. From frames 83–119 only five unique
  16-code vectors remain. Every VQ layer participates; VQ0..15 have 77–94
  adjacent repeats each. This is not isolated to a late residual VQ layer.
- The controller-truncated v2 ASR contains only the formula-like prefix and not
  the expected Chinese semantic tail (`content_pass=false`). The available data
  has no forced word/frame alignment, so it cannot prove the exact spoken word
  lost at frame 32. It does prove that the trajectory collapses before the
  68-frame teacher boundary and that increasing the global acoustic coefficient
  fourfold did not move it.

## Root-cause judgment

Uniform acoustic CE is reaching the global LoRA (all heads/local transformer are
frozen), but five steps plus PCGrad produce only a very small perturbation. The
same greedy text and all 16 code choices show that v2 did not alter content at
inference. The missing second half is best localized to the shared autoregressive
state entering a whole-frame repetition attractor around frame 32, not to EOS and
not to one VQ refinement layer. Teacher-forced CE hides this because it always
conditions on the correct preceding teacher codes.

## Next single variable

Do **not** raise acoustic total weight to 1.0. Keep v2's total `0.5`, five steps,
LoRA surface, PCGrad protectors and decoding fixed. First obtain a forced
alignment for each authorized teacher that marks the start of the Chinese
semantic tail. Change only the within-utterance acoustic weighting: give frames
from that audited boundary through teacher EOS 2x weight and renormalize the
per-utterance VQ loss so its total remains 0.5. This directly tests whether the
uniform objective under-trains the region that disappears, without increasing
global voice-drift pressure. Fail closed if alignment is missing, if any early
formula frame regresses, or if the free-run first repetition onset does not move
later while the semantic tail becomes complete.
