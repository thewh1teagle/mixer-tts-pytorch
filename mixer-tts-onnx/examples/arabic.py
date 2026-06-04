"""Arabic Mixer-TTS ONNX example.

Install the ONNX wrapper and Arabic phonemizer:

    uv pip install mixer-tts-onnx==0.1.0
    uv pip install "phoonnx @ git+https://github.com/TigreGotico/phoonnx.git@0700a982ae29f0228b52e1588849ce5aec19be85"

`phoonnx` should resolve to version 1.3.5a1 for this example.

Download the Arabic embedded-Vocos fp32 model:

    wget -O arabic_tts_v1_best_vocos_fp32.onnx https://huggingface.co/thewh1teagle/mixer-tts/resolve/main/arabic_tts_v1_best_vocos_fp32.onnx

The ONNX model metadata contains the Arabic IPA normalization replacements.
"""

from pathlib import Path

from mixer_tts_onnx import MixerTTS
from phoonnx.config import Alphabet
from phoonnx.phonemizers.ar import MantoqPhonemizer


def arabic_to_ipa(text: str, tts: MixerTTS) -> tuple[str, str]:
    phonemizer = MantoqPhonemizer(alphabet=Alphabet.IPA)
    vocalized = phonemizer.add_diacritics(text, "ar")
    ipa = tts.normalize_arabic_ipa(phonemizer.phonemize_string(vocalized, "ar"))
    return vocalized, ipa


def main() -> None:
    model_path = Path(__file__).resolve().parent / "arabic_tts_v1_best_vocos_fp32.onnx"
    out = Path(__file__).resolve().parent / "arabic_eval.wav"
    tts = MixerTTS(model_path)

    text = "في صباح هادئ، اكتشف المهندس بابا جديدا للذكاء الاصطناعي."
    vocalized, ipa = arabic_to_ipa(text, tts)
    print(f"text: {text}")
    print(f"vocalized: {vocalized}")
    print(f"ipa: {ipa}")

    tts.create(
        ipa,
        is_phonemes=True,
        output_path=out,
        speed=1.0,
    )
    print(out)


if __name__ == "__main__":
    main()
