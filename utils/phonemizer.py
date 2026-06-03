import torch
import phonemizer
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper

from models.symbols import symbols_to_id

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

global_phonemizer_en = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

def text_to_ipa(text: str):
    return global_phonemizer_en.phonemize([text])[0] 

def text_to_ids_en(text: str):
    phonemes_ipa = ' ' + global_phonemizer_en.phonemize([text])[0] + ' '
    phonemes_ids = [symbols_to_id[s] for s in phonemes_ipa if s in symbols_to_id]
    phonemes_ids = torch.LongTensor(phonemes_ids)
    return phonemes_ids
