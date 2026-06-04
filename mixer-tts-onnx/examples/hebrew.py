"""
Hebrew phoneme Mixer-TTS ONNX example.

Setup:
    cd mixer-tts-onnx
    uv run --package mixer-tts-onnx examples/hebrew.py

Download model:
    cd mixer-tts-onnx
    mkdir -p ../checkpoints/onnx
    wget https://huggingface.co/thewh1teagle/mixer-tts/resolve/main/synthetic_ft_80_best_vocos_int8.onnx -O ../checkpoints/onnx/synthetic_ft_80_last_vocos_int8.onnx

The example expects an embedded-vocoder ONNX model at:
    ../checkpoints/onnx/synthetic_ft_80_last_vocos_int8.onnx
"""

from pathlib import Path

from mixer_tts_onnx import MixerTTS


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "checkpoints" / "onnx" / "synthetic_ft_80_last_vocos_int8.onnx"
    tts = MixerTTS(model_path)
    out = Path(__file__).resolve().parent / "hebrew_eval.wav"
    tts.create(
        "miχtavˈim jeχolˈim leʃanˈot histˈoʁja, lehatsˈit milχamˈot ʔˈo lehavˈi ʃalˈom.",
        is_phonemes=True,
        output_path=out,
        speed=1.0,
    )
    print(out)


if __name__ == "__main__":
    main()
