# Same-speaker single-train pilot (preflight contract)

## Purpose

The next experiment changes one causal variable: teacher target and explicit
reference are both Junhao. It does not reuse the failed cross-speaker tail
weighting experiment.

## Frozen experiment

- Train: exactly one approved Junhao professional utterance (`Ca(OH)2`).
- Heldout: exactly one structurally different approved Junhao utterance
  (`CoCl2·6H2O`). It is evaluation-only and cannot appear in train or PCGrad.
- Protection: exactly five natural Junhao utterances, acoustic-only PCGrad.
  No professional protector is allowed because it could leak heldout content.
- Reference: one additional independent Junhao natural utterance. Its exact
  `ref_audio_codes` and hashes must be identical in train, heldout and protector
  rows. It cannot be any target/protector utterance.
- Human gate: every target and the reference require speaker **and** content
  approval in one immutable JSON report; the bundle supplies its SHA-256.
- Objective: sequence-balanced text weight 1 plus total acoustic weight 0.125,
  evenly divided across 16 VQ heads. This is the smallest previously reviewed
  nonzero acoustic objective and isolates speaker/reference alignment. Tail
  weighting and a weight increase are forbidden.
- Adapter: fresh global AR attention+MLP LoRA only; frozen clean base, no
  stacking and no modules-to-save.
- Budget: at most four optimizer steps. Persist/evaluate step 1, 2 and 4;
  choose the earliest candidate passing content, speaker and natural-retention
  gates. A failed earlier checkpoint does not authorize changing the objective.

## Asset preflight status

The immutable machine preflight is bound by SHA-256
`dc8207757d9662a14ce64dfe533b0d0e434512133493d9541219004a598874fb`.
It binds the source manifest, A/B codec and gate artifacts, historical human
ledger, independent `junhao_real_a03` reference, and five natural protectors.
The legacy `pass_user` string is never trusted as a gate. Training remains
blocked until a new bundle embeds the exact reference codes and passes this
validator.

`same_speaker_pilot.validate_same_speaker_pilot` is the fail-closed bundle gate.
It is deliberately not wired into `sft.py` until real assets pass preflight, so
no partially valid command can launch training.
