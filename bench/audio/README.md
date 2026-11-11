Place generated and recorded WAV fixtures here.

- `tts-sample.wav` is written by `bench/probe/cartesia_ttfa.py --save-audio`.
- `recorded/` is filled by a real-mic session with `ASR_DUMP_DIR` set; score
  it with `bench/probe/dump_wer.py` (reading script and procedure: runbook §6,
  "Capturing reference recordings"). Those recordings are also the only
  real-speech input for `bench/probe/vad_accuracy.py`.
