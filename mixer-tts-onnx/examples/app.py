"""
Hebrew TTS Gradio app using Mixer-TTS ONNX + renikud-onnx.

Setup:
    cd mixer-tts-onnx
    uv run --package mixer-tts-onnx \
        --with "gradio==6.16.0" \
        --with "renikud-onnx==0.1.0" \
        --with "num2words==0.5.14" \
        examples/app.py

Download models:
    cd mixer-tts-onnx
    wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O ../renikud.onnx
    wget https://huggingface.co/thewh1teagle/mixer-tts/resolve/main/synthetic_ft_80_best_vocos_int8.onnx -O synthetic_ft_80_last_vocos_int8.onnx

The Mixer-TTS model defaults to:
    synthetic_ft_80_last_vocos_int8.onnx
"""

from __future__ import annotations

import re
from pathlib import Path

import espeakng_loader
import gradio as gr
import numpy as np
from num2words import num2words
from phonemizer import phonemize as phonemize_en
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from renikud_onnx import G2P

from mixer_tts_onnx import MixerTTS

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

DEFAULT_SPEED = 0.9
EXAMPLE_TEXT = "שימו לב נוסעים יקרים, הרכבת תכנס לתחנת תל אביב מרכז בעוד מספר דקות."
EXAMPLE_PHONEMES = "miχtavˈim jeχolˈim leʃanˈot histˈoʁja, lehatsˈit milχamˈot ʔˈo lehavˈi ʃalˈom."

LATIN_WORD_RE = re.compile(r"[a-zA-Z]+")
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
RENIKUD_PATH = REPO_ROOT / "renikud.onnx"
MIXER_MODEL_PATH = PACKAGE_ROOT / "synthetic_ft_80_last_vocos_int8.onnx"

g2p = G2P(RENIKUD_PATH)
tts = MixerTTS(MIXER_MODEL_PATH)


def expand_numbers(text: str) -> str:
    def replace_number(match: re.Match) -> str:
        number = match.group().replace(",", "")
        if "." in number:
            whole, fraction = number.split(".", maxsplit=1)
            whole_words = num2words(int(whole), lang="he")
            fraction_words = " ".join(num2words(int(digit), lang="he") for digit in fraction)
            return f"{whole_words} נקודה {fraction_words}"
        return num2words(int(number), lang="he")

    return NUMBER_RE.sub(replace_number, text)


def to_phonemes(text: str) -> str:
    text = text.replace("\n", " ")
    text = expand_numbers(text)

    def replace_latin(match: re.Match) -> str:
        return phonemize_en(
            match.group(0),
            backend="espeak",
            language="en-us",
            strip=True,
            with_stress=True,
        ).strip()

    return g2p.phonemize(LATIN_WORD_RE.sub(replace_latin, text))


def synthesize(phonemes_fn, text: str, speed: float = DEFAULT_SPEED) -> tuple[str, tuple[int, np.ndarray]]:
    phonemes = phonemes_fn(text).strip()
    if not phonemes:
        return "", (tts.sample_rate, np.zeros(1, dtype=np.float32))

    samples = tts.create(phonemes, is_phonemes=True, speed=speed)
    return phonemes, (tts.sample_rate, samples)


def synthesize_text(text: str, speed: float):
    return synthesize(to_phonemes, text, speed)


def synthesize_phonemes(phonemes: str, speed: float):
    return synthesize(lambda value: value, phonemes, speed)


with gr.Blocks(title="Hebrew Mixer-TTS") as demo:
    gr.Markdown("# Hebrew Mixer-TTS")

    with gr.Tabs():
        with gr.Tab("Text"):
            text_input = gr.Textbox(
                label="Hebrew text",
                placeholder="הקלד טקסט בעברית...",
                lines=4,
                rtl=True,
                value=EXAMPLE_TEXT,
            )
            text_speed = gr.Slider(
                label="Speed",
                minimum=0.75,
                maximum=1.35,
                value=DEFAULT_SPEED,
                step=0.01,
            )
            text_btn = gr.Button("Generate", variant="primary")
            text_phonemes_out = gr.Textbox(label="IPA phonemes", interactive=False, lines=2)
            text_audio_out = gr.Audio(label="Audio", type="numpy", autoplay=True)

            text_btn.click(
                synthesize_text,
                inputs=[text_input, text_speed],
                outputs=[text_phonemes_out, text_audio_out],
            )

        with gr.Tab("Phonemes"):
            phonemes_input = gr.Textbox(
                label="IPA phonemes",
                placeholder="e.g. ʃalˈom ʕolˈam",
                lines=4,
                value=EXAMPLE_PHONEMES,
            )
            phonemes_speed = gr.Slider(
                label="Speed",
                minimum=0.75,
                maximum=1.35,
                value=DEFAULT_SPEED,
                step=0.01,
            )
            phonemes_btn = gr.Button("Generate", variant="primary")
            phonemes_out = gr.Textbox(label="IPA phonemes", interactive=False, lines=2)
            phonemes_audio_out = gr.Audio(label="Audio", type="numpy", autoplay=True)

            phonemes_btn.click(
                synthesize_phonemes,
                inputs=[phonemes_input, phonemes_speed],
                outputs=[phonemes_out, phonemes_audio_out],
            )


if __name__ == "__main__":
    demo.launch()
