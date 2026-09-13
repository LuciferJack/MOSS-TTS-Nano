# MOSS-TTS Nano stop-loss audit

## Label path

`MossTTSNanoSFTDataset` emits one text-channel
`audio_assistant_slot_token_id` target for every codec frame and appends exactly
one `audio_end_token_id` target. Audio targets on the final row are padding and
are excluded from every VQ loss. The runtime makes the same binary decision:
slot means continue and audio-end means stop.

The audited model configuration uses slot ID `9`, audio-end ID `7`, and audio
padding ID `1024`. The four timing teachers contain 35, 42, 45 and 53 frames.
Consequently, legacy mean CE gives the only stop target just `1/(T+1)`, or
1.85%–2.78%, of the text-channel token mass.

## Objectives

- `--eos-loss-mode token_weight` is the compatibility default. With
  `--eos-loss-weight 1`, it is exactly the former full token mean. Other weights
  multiply only the audio-end token before the weighted mean.
- `--eos-loss-mode sequence_balanced` computes each sample's continuation CE
  mean and its single stop CE independently, combines them as
  `(continue_mean + weight * stop_loss) / (1 + weight)`, then averages samples.
  This prevents long utterances from reducing stop supervision. The first
  controlled experiment should keep weight `1` for an equal 50/50 objective.

Both modes fail closed unless every sample has exactly one audio-end target and
at least one continuation target. Unexpected supervised text targets also fail.
Dataset packing separately refuses to truncate any sequence that would lose its
audio-end target.

## Diagnostics and CPU evidence

Diagnostics expose continuation/stop loss, count, runtime-binary accuracy, and
signed slot-vs-end margin, plus sequence count. Accuracy deliberately does not
use full-vocabulary argmax because generation termination compares slot and
audio-end.

On Contabo CPU, the focused loss/packing/protected-gradient suite passes 19/19,
and the complete finetuning test discovery passes 93/93.
The real four-teacher/eight-protector bundle packs to at most 160 tokens at
`max_length=256` (teacher maximum 147, protector maximum 160); no stop target is
truncated. No training or GPU execution was performed for this change.
