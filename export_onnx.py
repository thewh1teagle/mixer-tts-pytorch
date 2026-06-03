from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from vocos import Vocos

from models import MixerTTSModel
from models.symbols import symbols


ROOT = Path(__file__).resolve().parent


class MixerTTSOnnx(torch.nn.Module):
    def __init__(self, model: MixerTTSModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        token_ids: torch.LongTensor,
        pace: torch.Tensor,
        speaker: torch.Tensor,
        emotion: torch.Tensor,
        pitch_mul: torch.Tensor,
        pitch_add: torch.Tensor,
    ) -> torch.Tensor:
        def pitch_transform(pitch_pred, enc_mask_sum, mean, std):
            return pitch_mul * pitch_pred + pitch_add

        return self.model.infer(
            token_ids,
            pace=pace,
            pitch_transform=pitch_transform,
            speaker=speaker,
            emotion=emotion,
        ).transpose(1, 2)


class MixerTTSVocosOnnx(torch.nn.Module):
    def __init__(self, acoustic: MixerTTSOnnx, vocoder: Vocos) -> None:
        super().__init__()
        self.acoustic = acoustic
        self.backbone = vocoder.backbone
        self.out = vocoder.head.out
        self.n_fft = vocoder.head.istft.n_fft
        self.hop_length = vocoder.head.istft.hop_length
        self.trim = (self.n_fft - self.hop_length) // 2

        freq_bins = self.n_fft // 2 + 1
        n = torch.arange(self.n_fft, dtype=torch.float32)
        k = torch.arange(freq_bins, dtype=torch.float32)[:, None]
        angle = 2 * math.pi * k * n[None, :] / self.n_fft
        real_basis = torch.cos(angle)
        imag_basis = -torch.sin(angle)
        real_basis[1:-1] *= 2.0
        imag_basis[1:-1] *= 2.0
        basis = torch.cat([real_basis, imag_basis], dim=0) / self.n_fft
        self.register_buffer("irfft_basis", basis)

        window = vocoder.head.istft.window.detach().float()
        self.register_buffer("window", window)
        ola_weight = torch.zeros(self.n_fft, 1, self.n_fft)
        for i in range(self.n_fft):
            ola_weight[i, 0, i] = window[i]
        self.register_buffer("ola_weight", ola_weight)
        env_weight = torch.zeros(self.n_fft, 1, self.n_fft)
        window_sq = window.square()
        for i in range(self.n_fft):
            env_weight[i, 0, i] = window_sq[i]
        self.register_buffer("env_weight", env_weight)

    def istft_real(self, spec: torch.Tensor) -> torch.Tensor:
        log_mag, phase = torch.chunk(spec, 2, dim=1)
        mag = torch.exp(log_mag).clamp(max=100.0)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        packed = torch.cat([real, imag], dim=1).transpose(1, 2)
        frames = torch.matmul(packed, self.irfft_basis).transpose(1, 2)
        y = torch.nn.functional.conv_transpose1d(frames, self.ola_weight, stride=self.hop_length)
        envelope_input = torch.ones_like(frames)
        envelope = torch.nn.functional.conv_transpose1d(envelope_input, self.env_weight, stride=self.hop_length)
        y = y / envelope.clamp_min(1e-8)
        return y[:, 0, self.trim : -self.trim]

    def forward(
        self,
        token_ids: torch.LongTensor,
        pace: torch.Tensor,
        speaker: torch.Tensor,
        emotion: torch.Tensor,
        pitch_mul: torch.Tensor,
        pitch_add: torch.Tensor,
    ) -> torch.Tensor:
        mel = self.acoustic(token_ids, pace, speaker, emotion, pitch_mul, pitch_add)
        features = self.backbone(mel)
        spec = self.out(features).transpose(1, 2)
        return self.istft_real(spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "pretrained" / "mixer_lj_80.pth"))
    parser.add_argument("--output", default=str(ROOT / "checkpoints" / "onnx" / "mixer_lj_80.onnx"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--with-vocoder", action="store_true")
    parser.add_argument("--vocoder", default="BSC-LT/vocos-mel-22khz")
    parser.add_argument("--no-int8", action="store_true", help="Export fp32 ONNX instead of default int8 dynamic quantization")
    return parser.parse_args()


def remove_sidecar(output: Path) -> None:
    sidecar = output.with_suffix(output.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()


def add_metadata(output: Path, metadata: dict[str, str]) -> None:
    onnx_model = onnx.load(output.as_posix(), load_external_data=False)
    onnx.checker.check_model(onnx_model)
    del onnx_model.metadata_props[:]
    for key, value in metadata.items():
        prop = onnx_model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(onnx_model, output.as_posix(), save_as_external_data=False)


def materialize_module_tensors(module: torch.nn.Module) -> None:
    for child in module.children():
        materialize_module_tensors(child)
    for name, param in list(module._parameters.items()):
        if param is not None:
            module._parameters[name] = torch.nn.Parameter(param.detach().clone(), requires_grad=False)
    for name, buffer in list(module._buffers.items()):
        if buffer is not None:
            module._buffers[name] = buffer.detach().clone()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = MixerTTSModel(**ckpt["net_config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    acoustic = MixerTTSOnnx(model).to(args.device).eval()
    if args.with_vocoder:
        vocoder = Vocos.from_pretrained(args.vocoder).to(args.device).eval()
        materialize_module_tensors(vocoder)
        wrapper = MixerTTSVocosOnnx(acoustic, vocoder).to(args.device).eval()
        output_names = ["audio"]
        dynamic_axes = {
            "token_ids": {0: "batch", 1: "token"},
            "audio": {0: "batch", 1: "sample"},
        }
        output_type = "audio"
    else:
        wrapper = acoustic
        output_names = ["mel_spec"]
        dynamic_axes = {
            "token_ids": {0: "batch", 1: "token"},
            "mel_spec": {0: "batch", 2: "frame"},
        }
        output_type = "mel"
    sample = (
        torch.LongTensor([[1, 2, 3]]).to(args.device),
        torch.FloatTensor([1.0]).to(args.device),
        torch.IntTensor([0]).to(args.device),
        torch.IntTensor([0]).to(args.device),
        torch.FloatTensor([1.0]).to(args.device),
        torch.FloatTensor([0.0]).to(args.device),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fp32_output = output if args.no_int8 else output.with_suffix(".fp32.tmp.onnx")
    remove_sidecar(fp32_output)
    torch.onnx.export(
        wrapper,
        sample,
        fp32_output.as_posix(),
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
        opset_version=args.opset,
        input_names=["token_ids", "pace", "speaker", "emotion", "pitch_mul", "pitch_add"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    remove_sidecar(fp32_output)

    metadata = {
        "model_type": "mixer-tts",
        "checkpoint": Path(args.checkpoint).name,
        "sample_rate": "22050",
        "mel_channels": "80",
        "output_type": output_type,
        "vocoder": args.vocoder if args.with_vocoder else "",
        "symbols": "".join(symbols),
        "phonemizer": "espeak",
        "phonemizer_language": "en-us",
        "quantization": "fp32" if args.no_int8 else "int8_dynamic",
        "inputs_json": json.dumps(
            {
                "token_ids": "int64[batch,tokens]",
                "pace": "float32[1], higher is slower",
                "speaker": "int32[1]",
                "emotion": "int32[1]",
                "pitch_mul": "float32[1]",
                "pitch_add": "float32[1]",
            }
        ),
    }
    add_metadata(fp32_output, metadata)
    if args.no_int8:
        final_output = fp32_output
    else:
        remove_sidecar(output)
        quantize_dynamic(
            model_input=fp32_output.as_posix(),
            model_output=output.as_posix(),
            op_types_to_quantize=["MatMul", "Gemm"],
            weight_type=QuantType.QInt8,
        )
        add_metadata(output, metadata)
        fp32_output.unlink(missing_ok=True)
        final_output = output
    remove_sidecar(output)
    print(final_output)


if __name__ == "__main__":
    main()
