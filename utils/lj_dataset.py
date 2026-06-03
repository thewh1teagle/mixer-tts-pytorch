import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

from models.symbols import symbols_to_id
from models.mixer_tts.modules.data_function import BetaBinomialInterpolator
from utils.audio import MelSpectrogram, rms_normalize, fft_filter
from utils import read_lines_from_file


def normalize_pitch(pitch, 
                    mean=211.37922361962185, 
                    std=53.725665480075286
                    ):
    zeros = (pitch == 0.0)
    pitch -= mean
    pitch /= std
    pitch[zeros] = 0.0
    return pitch



class LJDataset(Dataset):
    def __init__(self,
                 audio_dir:str='I:/tts/english/LJSpeech-1.1/wavs',
                 pitch_dir:str='./data/lj_pitches/wavs',
                 textfile_path:str='./data/lj-meta-phon.csv',
                 sr_target:int=22050,
                 rms_db:float=-27,
                 f_cutoff:float=60, 
                 f0_mean:float=211.37922361962185,
                 f0_std:float=53.725665480075286,
                 ):
        
        """
        
        Returns (phonemes_ids, mel_log, phonemes_ids_length, pitch_mel, energy, 
                speaker, emotion, attn_prior, audio_filepath)
        """
        super().__init__()

        self.mel_fn = MelSpectrogram()
        self.audio_dir = audio_dir
        self.pitch_dir = pitch_dir
        self.sr_target = sr_target
        self.rms_db = rms_db
        self.f_cutoff = f_cutoff
        self.f0_mean = f0_mean
        self.f0_std = f0_std
        
        self.betabinomial_interpolator = BetaBinomialInterpolator()
        
        self.data = []
        self._load_textfile(textfile_path)
        
    def _load_textfile(self, textfile_path):
        lines = read_lines_from_file(textfile_path)

        audio_ids, audio_id_to_phones = [], {}
        for line in lines:
            audio_id, _, phones = line.split('|')
            audio_ids.append(audio_id)
            audio_id_to_phones[audio_id] = phones
        self.data = audio_ids
        self.audio_id_to_phones = audio_id_to_phones        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_id = self.data[idx]       
        
        audio_filepath = os.path.join(self.audio_dir, audio_id) + '.wav'

        wave, sr = torchaudio.load(audio_filepath)
        
        if self.f_cutoff is not None:
            wave = fft_filter(wave, self.sr_target, self.f_cutoff)
        if self.rms_db is not None:
            wave = rms_normalize(wave, self.rms_db)
        
        phonemes_ipa = ' ' + self.audio_id_to_phones[audio_id] + ' '
        phonemes_ids = [symbols_to_id[s] for s in phonemes_ipa if s in symbols_to_id]
        phonemes_ids = torch.LongTensor(phonemes_ids)
  
        mel_raw = self.mel_fn(wave)        
        mel_log = mel_raw.clamp_min(1e-5).log().squeeze()

        pitch_filepath = os.path.join(self.pitch_dir, audio_id) + '.wav.pt'
        pitch_mel = torch.load(pitch_filepath)
        pitch_mel = normalize_pitch(pitch_mel, self.f0_mean, self.f0_std)[None]
        
        energy = torch.norm(mel_log.float(), dim=0, p=2)
        attn_prior = torch.from_numpy(
            self.betabinomial_interpolator(mel_log.size(1), len(phonemes_ids)))

        speaker = 0
        emotion = 0
        return (phonemes_ids, mel_log, len(phonemes_ids), pitch_mel, energy, 
                speaker, emotion, attn_prior, audio_filepath)
        

class DynBatchDataset(LJDataset):
    def __init__(self,
                 audio_dir:str='I:/tts/english/LJSpeech-1.1/wavs',
                 pitch_dir:str='./data/lj_pitches/wavs',
                 textfile_path:str='./data/lj-meta-phon.csv', 
                 sr_target:int=22050,
                 rms_db:float=-27,
                 f_cutoff:float=60, 
                 f0_mean:float=211.37922361962185,
                 f0_std:float=53.725665480075286,
                 ):
        super().__init__(
            audio_dir=audio_dir, 
            pitch_dir=pitch_dir, 
            textfile_path=textfile_path,  
            sr_target=sr_target,
            rms_db=rms_db,
            f_cutoff=f_cutoff,
            f0_mean=f0_mean,
            f0_std=f0_std,
        )
        
        self.max_id_lens = [0, 50, 100, 160, 210, 300, 5000]
        self.batch_sizes = [32, 16, 10, 8, 6, 4]

        self.id_batches = []
        self.shuffle()

    def shuffle(self):
      
        ids_lens = [
            len(self.audio_id_to_phones[audio_id]) for audio_id in self.data]

        ids_per_bs = {b: [] for b in self.batch_sizes}

        for i, ids_len in enumerate(ids_lens):
            b_idx = next(i for i in range(len(self.max_id_lens)-1)
                         if self.max_id_lens[i] <= ids_len < self.max_id_lens[i+1])
            ids_per_bs[self.batch_sizes[b_idx]].append(i)

        id_batches = []

        for bs, ids in ids_per_bs.items():
            np.random.shuffle(ids)
            ids_chnk = [ids[i:i+bs] for i in range(0, len(ids), bs)]
            id_batches += ids_chnk

        self.id_batches = id_batches

    def __len__(self):
        return len(self.id_batches)

    def __getitem__(self, idx):
        batch = [super(DynBatchDataset, self).__getitem__(idx)
                 for idx in self.id_batches[idx]]
        return batch
