# MOSS reference/codebook content-timbre audit

## Scope

Read-only CPU audit. No training, GPU, waveform decoding, or playback was used.
Evidence comes from the shipped MOSS-TTS-Nano model, its tokenizer README, five
authorized Doubao hydrate code sequences (351 frames), eight Junhao protection
sequences (239 frames), and saved v2/v3 Cu traces.

## Where reference audio enters the model

There is no separate speaker encoder or speaker embedding in MOSS-TTS-Nano.
`build_inference_input_ids` packs reference codec frames as ordinary
`audio_user_slot` rows. `_build_inputs_embeds` starts with the text/slot embedding
and **adds all 16 codebook embeddings into one vector**. That vector enters the
same global GPT used for text and generated history. Every generated frame is
packed as an `audio_assistant_slot` row and fed back by the identical sum.

For the next frame, therefore, global state cannot directly recover a named
"content half" and "voice half": all codebooks, speaker reference, linguistic
history, and slot identity have already been mixed by addition and global
self-attention.

The local GPT receives one global hidden vector, then the assistant-slot text
embedding. It predicts VQ0, embeds the sampled VQ0 token, predicts VQ1, and so
on through VQ15. Thus VQ0 has no same-frame audio predecessor; VQq depends on
all VQ0...VQ(q-1). All audio heads are individually tied to their corresponding
audio embedding tables. The tokenizer documents a 16-stage residual quantizer
with 1,024 entries per codebook and variable-rate prefix decoding. Residual order
supports "earlier = coarser reconstruction, later = residual detail"; neither
the implementation nor documentation assigns semantics exclusively to early
books or timbre exclusively to late books.

## Per-codebook evidence

Smoothed Jensen-Shannon divergence compares Doubao teacher and Junhao protector
token histograms. Teacher-forced/free columns use the v3 Cu target under a Junhao
prompt; ranks are among 1,024 entries over 68 frames.

| VQ | JS bits | TF top1 | TF mean rank | free top1 | free mean rank |
|---:|---:|---:|---:|---:|---:|
| 0 | .8037 | 15 | 34.37 | 0 | 431.40 |
| 1 | .7688 | 10 | 63.37 | 0 | 406.43 |
| 2 | .7258 | 4 | 58.65 | 0 | 446.79 |
| 3 | .7259 | 5 | 93.68 | 0 | 437.50 |
| 4 | .7412 | 0 | 121.04 | 1 | 481.46 |
| 5 | .7941 | 0 | 107.75 | 0 | 455.87 |
| 6 | .7232 | 2 | 111.35 | 0 | 608.04 |
| 7 | .7962 | 1 | 129.19 | 0 | 578.46 |
| 8 | .7859 | 2 | 133.09 | 0 | 525.91 |
| 9 | .7057 | 5 | 158.69 | 0 | 495.68 |
| 10 | .7889 | 2 | 152.56 | 0 | 558.85 |
| 11 | .8118 | 2 | 172.15 | 0 | 531.19 |
| 12 | .7698 | 2 | 106.63 | 0 | 426.94 |
| 13 | .7589 | 1 | 130.69 | 0 | 405.21 |
| 14 | .7471 | 1 | 128.26 | 0 | 486.78 |
| 15 | .7532 | 1 | 112.32 | 0 | 465.07 |

VQ0-VQ3 are easier under teacher forcing, consistent with an RVQ coarse-to-fine
ordering, but their speaker-distribution divergence is not smaller; VQ0 has the
second-highest JS value. Later books are not a clean "voice-only" block either:
their teacher targets remain poorly ranked and their JS values overlap the early
range. Histogram JS is content-confounded and cannot by itself label a factor,
but it decisively supplies no split point.

As a code-only perturbation, v2 and v3 are identical through frame 23. After
their first global-policy divergence, VQ0 still matches on 106/120 frames while
most other books match only 24-35/120 (VQ8: 71/120). This does not make VQ0 a
content channel: both runs are in the same repeated attractor, and the local
autoregressive chain amplifies a changed early token across later heads.

## Can content alone be distilled while preserving Junhao timbre?

Not by selecting a fixed subset of MOSS codebooks on current evidence. Training
only VQ0 would supervise a coarse residual stage that still differs strongly by
speaker/data distribution. Training early books also changes the summed frame
embedding returned to the global GPT, so it changes later-frame rhythm, duration,
and every later codebook distribution. Conversely, freezing late heads does not
freeze their outputs because their local/global hidden inputs still move.

The safe conclusion is **no architectural separation guarantee**. A content-only
distillation claim requires an empirical invariance test with the same utterance
spoken by both voices; the current rows are near-matched at best and cannot
identify content versus speaker factors.

Do not assume VQ0 is semantic, label VQ8-VQ15 as timbre, train a hand-picked
subset, raise its loss, or ship a hybrid-code candidate. Those are unsupported
factorization assumptions and can silently move Junhao's voice.

## Single minimal experiment

Collect or use one authorization-matched parallel formula already closest in the
corpus (`CoCl₂·6H₂O`) from Doubao and Junhao, normalize the spoken text exactly,
and obtain an independent short reference from each speaker. With frozen base
weights on CPU, run a code-domain 2 x 2 teacher-forced probe:

1. target speaker Doubao/Junhao x reference speaker Doubao/Junhao;
2. record per-frame ranks for all 16 heads and global-hidden displacement;
3. repeat while masking contiguous **prefixes** of residual books (1, 2, 4, 8,
   16), never arbitrary suffixes, because the codec supports prefix bitrates and
   the local decoder is ordered;
4. select no content subset unless a prefix improves same-text rank similarly
   for both voices while reference-speaker swapping changes that prefix little
   and changes the complementary residuals strongly.

This is an optimizer-free identifiability experiment, not another SFT run. If no
prefix meets the invariance criterion, content and timbre are entangled for this
use case; the next architecture must introduce an explicit content representation
(phonemes/text hidden states) and a separate Junhao speaker conditioner rather
than pretending the existing RVQ indices provide that split.
