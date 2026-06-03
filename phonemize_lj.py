# %%
import phonemizer
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper

from utils import read_lines_from_file, write_lines_to_file

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

phonemizer_en = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

# %%

filepath = 'I:/tts/english/LJSpeech-1.1/metadata.csv'
lines = read_lines_from_file(filepath)

lines_new = []

for line in lines:
    sent_id, sent, sent_norm = line.split('|')
    sent_phon, sent_norm_phon = phonemizer_en.phonemize([sent, sent_norm])
    lines_new.append(f"{sent_id}|{sent_phon}|{sent_norm_phon}")

write_lines_to_file('./data/lj-meta-phon.csv', lines_new)
