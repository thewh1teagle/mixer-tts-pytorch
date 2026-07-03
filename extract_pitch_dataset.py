import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import write_lines_to_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--pitch-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--rmvpe-model", required=True)
    parser.add_argument("--rmvpe-threshold", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mel_length(num_samples, n_fft=1024, hop_length=256):
    pad_length = int((n_fft - hop_length) / 2)
    return int((num_samples + 2 * pad_length - n_fft) // hop_length + 1)


def fit_length(values, length):
    values = values.float().flatten()
    if values.numel() == length:
        return values
    if values.numel() == 0:
        return torch.zeros(length, dtype=torch.float32)
    return F.interpolate(values[None, None], size=length, mode="linear", align_corners=False)[0, 0]


def extract_rmvpe(audio_ids, audio_dir, pitch_dir, sample_rate, args):
    from rmvpe_onnx import RMVPE

    rmvpe = RMVPE(args.rmvpe_model)

    total_count = 0
    total_sum = 0.0
    total_sumsq = 0.0
    progress = tqdm(total=len(audio_ids), desc="extracting pitch (rmvpe-onnx)")

    for audio_id in audio_ids:
        pitch_path = pitch_dir / f"{audio_id}.wav.pt"
        if pitch_path.exists() and not args.overwrite:
            pitch_mel = torch.load(pitch_path, map_location="cpu")
        else:
            audio_path = audio_dir / f"{audio_id}.wav"
            info = sf.info(audio_path)
            wav, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
            f0 = rmvpe.extract(
                np.asarray(wav, dtype=np.float32),
                sample_rate=sample_rate,
                threshold=args.rmvpe_threshold,
            )
            pitch_mel = fit_length(torch.as_tensor(f0, dtype=torch.float32), mel_length(info.frames))
            pitch_mel = torch.nan_to_num(pitch_mel, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            torch.save(pitch_mel, pitch_path)

        voiced = pitch_mel[pitch_mel > 1].double()
        total_count += voiced.numel()
        total_sum += voiced.sum().item()
        total_sumsq += voiced.square().sum().item()
        progress.update(1)

    progress.close()
    return total_count, total_sum, total_sumsq


def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    pitch_dir = Path(args.pitch_dir)
    pitch_dir.mkdir(parents=True, exist_ok=True)

    audio_ids = [
        line.split("|", 1)[0]
        for line in Path(args.metadata).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    total_count, total_sum, total_sumsq = extract_rmvpe(
        audio_ids, audio_dir, pitch_dir, args.sample_rate, args
    )

    if total_count:
        mean = total_sum / total_count
        variance = max(0.0, total_sumsq / total_count - mean * mean)
        std = variance ** 0.5
    else:
        mean, std = 0.0, 1.0

    write_lines_to_file(
        path=os.path.join(pitch_dir, "mean_std.txt"),
        lines=[f"mean: {mean}", f"std: {std}"],
    )
    print(f"pitch files: {len(audio_ids)}")
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == "__main__":
    main()
