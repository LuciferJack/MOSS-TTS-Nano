# Formula-scoped duration/stop controller

Round 3 exhausted the permitted DAgger updates, so this path freezes MOSS and
all accepted adapters. It does not create Round 4. The controller consumes the
exact canonical text, exact spoken text, current frame index and the two runtime
text-channel logits (`assistant_slot`, `audio_end`). Hidden states are deliberately
excluded from the first prototype so the decision is reproducible on CPU.

Five independently authorized teacher boundaries define conservative duration
windows. Text features describe formula structure and utterance length; nearest
teacher prediction receives a one-sided guard equal to the worst inner
leave-one-out under-prediction. The controller may emit EOS only at this safe
upper edge, never at the predicted center. Leave-one-formula-out reports
`early_cut` and `before_cap` for every family rather than averaging them away.

Runtime priority is strict:

1. If MOSS selects native EOS, stop immediately; the controller cannot veto it.
2. Non-formula text, out-of-distribution/low-confidence text, non-finite logits,
   or frames below the conservative window cannot be overridden.
3. A continuing MOSS may be stopped only at the safe edge, below the hard cap,
   for a high-confidence formula query.

This is not a content-completion oracle. Before App integration, every LOFO fold
must have `early_cut=false`, all known formula traces must stop below cap, and
audio/content gates must confirm the final spoken token and Chinese semantic tail
are complete. A single early cut, missing per-frame trajectory, acronym/base
route activation, or content-tail failure disables controller deployment. The
current Round-3 artifact records only a 120-frame summary and lacks per-frame
slot/end logits, so it is insufficient to claim effectiveness; real traces must
be exported and hashed before evaluation.
