---
name: mixer-tts-1m-finetune
description: Fine-tune this mixer-tts-pytorch repo's 1.74M parameter Mixer-TTS checkpoint on an IPA-phonemized dataset, including uv setup, CUDA checks, checkpoint download, RMVPE pitch extraction, smoke testing, and infinite training with best/last checkpoint saves.
---

# Mixer-TTS 1M Fine-Tune

Use this workflow to fine-tune the small LJSpeech Mixer-TTS checkpoint (`mixer_lj_80.pth`, about 1.74M params) on an IPA-phonemized dataset.

## Prerequisites

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo and enter it:

```bash
git clone <repo-url> mixer-tts-pytorch
cd mixer-tts-pytorch
```

Install dependencies:

```bash
uv sync
```

Confirm PyTorch sees CUDA:

```bash
uv run python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
PY
```

## Download The 1M Checkpoint

Download pretrained Mixer-TTS checkpoints:

```bash
uv run python download_files.py
```

For this fine-tune, the required checkpoint is:

```text
pretrained/mixer_lj_80.pth
```

## Dataset Shape

Prepare a dataset directory like:

```text
data/synthetic-multilingual-speech-ft/
├── metadata.csv
└── wav/
    ├── sample-id-1.wav
    └── sample-id-2.wav
```

`metadata.csv` must be repo-compatible, pipe-delimited, 3 columns:

```text
audio_id|text|ipa_phones
```

Rules:

- `audio_id` maps to `wav/audio_id.wav`.
- `text` is ignored by the training loader but should contain the transcript.
- `ipa_phones` must contain only symbols present in `models/symbols.py`.
- WAVs should be mono, `22050 Hz`, preferably `PCM_16`.

Validate quickly:

```bash
uv run python - <<'PY'
from pathlib import Path
from models.symbols import symbols_to_id
root = Path("data/synthetic-multilingual-speech-ft")
missing = {}
rows = root.joinpath("metadata.csv").read_text(encoding="utf-8").splitlines()
for line in rows:
    audio_id, _, ipa = line.split("|")
    assert root.joinpath("wav", f"{audio_id}.wav").exists(), audio_id
    for ch in " " + ipa + " ":
        if ch not in symbols_to_id:
            missing[ch] = missing.get(ch, 0) + 1
print("rows:", len(rows))
print("missing_symbols:", missing)
PY
```

## Extract RMVPE Pitch

This repo uses `extract_pitch_dataset.py` to create Mixer-TTS-compatible pitch files:

```text
pitch/audio_id.wav.pt
pitch/mean_std.txt
```

Run:

```bash
uv run python extract_pitch_dataset.py \
  --audio-dir ./data/synthetic-multilingual-speech-ft/wav \
  --metadata ./data/synthetic-multilingual-speech-ft/metadata.csv \
  --pitch-dir ./data/synthetic-multilingual-speech-ft/pitch \
  --sample-rate 22050 \
  --rmvpe-root ../zero-tts \
  --rmvpe-device cuda \
  --rmvpe-batch-size 64 \
  --overwrite
```

The extractor writes dataset-specific F0 stats. Training reads these automatically.

## Smoke Test

Before starting an infinite run, train exactly one step, eval one fast batch, save `last.pth` and `best.pth`, then exit:

```bash
uv run python train_ft.py \
  --config ./configs/synthetic-ft-80.yaml \
  --max-steps 1 \
  --eval-max-batches 1
```

Expected output includes:

```text
iter=0 ... loss=...
iter=0 eval_loss=...
saved new best: ...
reached max_steps=1; exiting
```

Check:

```bash
ls -lh checkpoints/synthetic-ft-80/last.pth checkpoints/synthetic-ft-80/best.pth
```

## Infinite Training

Start training:

```bash
uv run python train_ft.py --config ./configs/synthetic-ft-80.yaml
```

In tmux pane 0:

```bash
tmux send-keys -t 4:0.0 "cd /path/to/mixer-tts-pytorch && uv run python train_ft.py --config ./configs/synthetic-ft-80.yaml 2>&1 | tee -a logs/synthetic-ft-80/train.log" C-m
```

Training behavior:

- Restores model weights from `pretrained/mixer_lj_80.pth`.
- Uses `data/synthetic-multilingual-speech-ft/metadata.csv`.
- Uses audio from `data/synthetic-multilingual-speech-ft/wav`.
- Uses RMVPE pitch files and `pitch/mean_std.txt`.
- Runs indefinitely until killed.
- Saves `checkpoints/synthetic-ft-80/last.pth` every `save_interval`.
- Runs eval every `eval_interval`.
- Saves `checkpoints/synthetic-ft-80/best.pth` when eval loss improves.

Default config:

```text
configs/synthetic-ft-80.yaml
```

Logs:

```text
logs/synthetic-ft-80/train.log
logs/synthetic-ft-80/
```
