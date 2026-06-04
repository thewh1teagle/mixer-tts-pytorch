from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import espeakng_loader
import numpy as np
import onnx
import onnxruntime as ort
import phonemizer
import soundfile as sf
from phonemizer.backend.espeak.wrapper import EspeakWrapper


def _metadata(path: str | Path) -> dict[str, str]:
    model = onnx.load(str(path), load_external_data=False)
    return {entry.key: entry.value for entry in model.metadata_props}


class MixerTTS:
    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = [provider for provider in providers if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.metadata = _metadata(self.model_path)
        symbols = self.metadata.get("symbols")
        if not symbols:
            raise ValueError(f"ONNX model has no 'symbols' metadata: {self.model_path}")
        sample_rate = self.metadata.get("sample_rate")
        if sample_rate is None:
            raise ValueError(f"ONNX model has no 'sample_rate' metadata: {self.model_path}")

        self.symbols = list(symbols)
        self.symbols_to_id = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.sample_rate = int(sample_rate)
        self.output_type = self.metadata.get("output_type", "mel")
        if self.output_type != "audio":
            raise ValueError(f"Expected embedded-vocoder ONNX with output_type=audio: {self.model_path}")
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language=self.metadata.get("phonemizer_language", "en-us"),
            preserve_punctuation=True,
            with_stress=True,
        )
        replacements_json = self.metadata.get("arabic_ipa_replacements_json")
        if replacements_json:
            self.arabic_ipa_replacements = json.loads(replacements_json)
        else:
            self.arabic_ipa_replacements = {}

    def text_to_ipa(self, text: str) -> str:
        return self.phonemizer.phonemize([text])[0]

    def normalize_arabic_ipa(self, ipa: str) -> str:
        for source, target in self.arabic_ipa_replacements.items():
            ipa = ipa.replace(source, target)
        return ipa

    def text_to_ids(self, text: str, *, is_phonemes: bool = False) -> np.ndarray:
        ipa = text if is_phonemes else self.text_to_ipa(text)
        return np.array(
            [[self.symbols_to_id[ch] for ch in f" {ipa} " if ch in self.symbols_to_id]],
            dtype=np.int64,
        )

    def _run(
        self,
        text: str,
        *,
        is_phonemes: bool = False,
        speed: float = 1.0,
        speaker: int = 0,
        emotion: int = 0,
        pitch_mul: float = 1.0,
        pitch_add: float = 0.0,
    ) -> np.ndarray:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        token_ids = self.text_to_ids(text, is_phonemes=is_phonemes)
        # Mixer-TTS uses pace: higher is slower. Expose user-facing speed instead.
        pace = np.array([1.0 / speed], dtype=np.float32)
        outputs = self.session.run(
            None,
            {
                "token_ids": token_ids,
                "pace": pace,
                "speaker": np.array([speaker], dtype=np.int32),
                "emotion": np.array([emotion], dtype=np.int32),
                "pitch_mul": np.array([pitch_mul], dtype=np.float32),
                "pitch_add": np.array([pitch_add], dtype=np.float32),
            },
        )
        return outputs[0].astype(np.float32)

    def create(
        self,
        text: str,
        *,
        is_phonemes: bool = False,
        speed: float = 1.0,
        output_path: str | Path | None = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        output = self._run(text, is_phonemes=is_phonemes, speed=speed, **kwargs)
        audio = output[0].astype(np.float32)
        if normalize:
            peak = np.maximum(np.abs(audio).max(), 1e-6)
            audio = audio / peak
        if output_path is not None:
            sf.write(str(output_path), audio, self.sample_rate, subtype="PCM_16")
        return audio


__all__ = ["MixerTTS"]
