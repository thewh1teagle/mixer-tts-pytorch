import argparse
import os
import sys
from pathlib import Path

import librosa
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
    parser.add_argument("--rmvpe-root", default="../zero-tts")
    parser.add_argument("--rmvpe-model", default=None)
    parser.add_argument("--rmvpe-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rmvpe-batch-size", type=int, default=32)
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


def batched_rmvpe(audio_ids, audio_dir, pitch_dir, sample_rate, args):
    zero_tts_root = Path(args.rmvpe_root).resolve()
    sys.path.insert(0, zero_tts_root.as_posix())
    from huggingface_hub import hf_hub_download
    from src.acoustic.rmvpe import RMVPE, SAMPLE_RATE as RMVPE_SAMPLE_RATE

    device = torch.device(args.rmvpe_device)
    model_path = args.rmvpe_model or hf_hub_download("stylish-tts/pitch_extractor", "rmvpe.safetensors")
    hop_length = RMVPE_SAMPLE_RATE // (sample_rate // 256)
    rmvpe = RMVPE(model_path, device=device, hop_length=hop_length)

    pending_by_length = {}
    for audio_id in audio_ids:
        pitch_path = pitch_dir / f"{audio_id}.wav.pt"
        if pitch_path.exists() and not args.overwrite:
            pitch_mel = torch.load(pitch_path, map_location="cpu")
            voiced = pitch_mel[pitch_mel > 1].double()
            pending_by_length.setdefault(None, []).append(
                (audio_id, None, pitch_mel, voiced.numel(), voiced.sum().item(), voiced.square().sum().item())
            )
            continue
        info = sf.info(audio_dir / f"{audio_id}.wav")
        pending_by_length.setdefault(info.frames, []).append((audio_id, info.frames, None, 0, 0.0, 0.0))

    existing = pending_by_length.pop(None, [])
    total_count = sum(x[3] for x in existing)
    total_sum = sum(x[4] for x in existing)
    total_sumsq = sum(x[5] for x in existing)

    progress = tqdm(total=len(audio_ids), desc=f"extracting pitch ({device} rmvpe)")
    if existing:
        progress.update(len(existing))

    for frame_count in sorted(pending_by_length):
        rows = pending_by_length[frame_count]
        for start in range(0, len(rows), args.rmvpe_batch_size):
            batch = rows[start:start + args.rmvpe_batch_size]
            waves = []
            for audio_id, _, *_ in batch:
                wav, _ = librosa.load(audio_dir / f"{audio_id}.wav", sr=sample_rate, mono=True)
                waves.append(torch.from_numpy(wav).float())
            audio = torch.stack(waves)
            with torch.inference_mode():
                f0_batch = rmvpe.infer_from_audio_batch(
                    audio,
                    sample_rate=sample_rate,
                    thred=args.rmvpe_threshold,
                )
            f0_batch = torch.as_tensor(f0_batch).float().cpu()
            for row, (audio_id, _, *__) in enumerate(batch):
                target_len = mel_length(frame_count)
                pitch_mel = fit_length(f0_batch[row], target_len)
                pitch_mel = torch.nan_to_num(pitch_mel, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                pitch_path = pitch_dir / f"{audio_id}.wav.pt"
                torch.save(pitch_mel, pitch_path)
                voiced = pitch_mel[pitch_mel > 1].double()
                total_count += voiced.numel()
                total_sum += voiced.sum().item()
                total_sumsq += voiced.square().sum().item()
            progress.update(len(batch))
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

    total_count, total_sum, total_sumsq = batched_rmvpe(
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
