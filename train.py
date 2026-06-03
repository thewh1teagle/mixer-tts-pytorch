# %%
import os
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.mixer_tts.mixer_tts import MixerTTSModel
from models.mixer_tts.modules.data_function import (TTSCollate, batch_to_gpu)
from models.mixer_tts import net_config
from models.common.loss import PatchDiscriminator, extract_chunks, calc_feature_match_loss
from models.symbols import symbols

from utils import get_config
from utils.training import save_states
from utils.lj_dataset import DynBatchDataset

import matplotlib.pyplot as plt

config = get_config('./configs/ljspeech-384.yaml')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# %%

train_dataset_dyn = DynBatchDataset(
    audio_dir=config.train_audio_dir,
    textfile_path=config.train_labels,
    pitch_dir=config.pitch_dir,    
    rms_db=None, f_cutoff=None,
    )

# %%

collate_fn = TTSCollate()
sampler, shuffle, drop_last = None, True, True
train_loader = DataLoader(train_dataset_dyn,
                          batch_size=1,
                          collate_fn=lambda x: collate_fn(x[0]),
                        #   collate_fn=collate_fn,
                          shuffle=shuffle, drop_last=drop_last,
                          sampler=sampler)


# (phonemes_ids, mel_log, phonemes_len, pitch_mel, 
# energy, speaker, attn_prior, fpath) = train_dataset[0]

# (text_padded, input_lengths, mel_padded, output_lengths, len_x,
#     pitch_padded, energy_padded, speaker, emotion, attn_prior_padded,
#     audiopaths) = next(iter(train_loader))

# next(iter(train_loader))

# %%

n = config.dim
n = [80, 128, 384][1]

net_config.update({
    'num_tokens': len(symbols),
    'padding_idx': 0,
    'symbols_embedding_dim': n,
    'n_speakers': 16, 'n_emotions': 16    
})

model = MixerTTSModel(**net_config)
model = model.to(device)

# %%

train_gan = True

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                              weight_decay=config.weight_decay)

optimizer.param_groups[0]['betas'] = (0.0, 0.99) if train_gan else (0.9, 0.98)


# %% Discriminator

critic = PatchDiscriminator(1, 32).to(device)

optimizer_d = torch.optim.AdamW(critic.parameters(),
                                lr=1e-4, betas=(0.0, 0.99),
                                weight_decay=config.weight_decay)
tar_len = 128

# %%
# resume from existing checkpoint
n_epoch, n_iter = 0, 0

config.restore_model = f'I:/checkpoints/lj/exp{n}-gan/states.pth'

if config.restore_model != '':
    state_dicts = torch.load(config.restore_model)

    model.load_state_dict(state_dicts['model'], strict=False)
    if 'optim' in state_dicts:
        try:optimizer.load_state_dict(state_dicts['optim'])
        except: print('Unable to load optimizer states!')
    if 'model_d' in state_dicts:
        critic.load_state_dict(state_dicts['model_d'], strict=False)
    if 'optim_d' in state_dicts:
        optimizer_d.load_state_dict(state_dicts['optim_d'])
    if 'epoch' in state_dicts:
        n_epoch = state_dicts['epoch']
    if 'iter' in state_dicts:
        n_iter = state_dicts['iter']

model.add_bin_loss = True
model.bin_loss_scale = 1.0

# %%

config.checkpoint_dir = f'I:/checkpoints/lj/exp{n}-gan'
if not os.path.exists(config.checkpoint_dir): os.makedirs(config.checkpoint_dir)

config.log_dir = f'logs/lj/exp{n}_5'
writer = SummaryWriter(config.log_dir)

# %% TRAINING LOOP

torch.cuda.empty_cache()

model.train()
critic.train()

for epoch in range(n_epoch, config.epochs):
    train_dataset_dyn.shuffle() 
    for batch in train_loader:
        
        (text_padded, input_lengths,
         mel_padded, output_lengths,
         pitch_padded, energy_padded,
         speaker, emotion,
         attn_prior_padded, audiopaths,
         ), y, _ = batch_to_gpu(batch)
        
        (pred_spect, _, 
         pred_log_durs, pred_pitch, pred_energy, 
         attn_soft, attn_logprob, attn_hard, attn_hard_dur,
         ) = model(
            text=text_padded,
            text_len=input_lengths,
            pitch=pitch_padded[:,0],
            energy=energy_padded,
            spect=mel_padded,
            spect_len=output_lengths,
            attn_prior=attn_prior_padded,
            lm_tokens=None,
            speaker=speaker,
            emotion=emotion,
        )

        if train_gan:
            tar_len_ = min(output_lengths.min(), tar_len)
            # extract chunks for critic
            ofx_perc = torch.rand(output_lengths.size()).to(device)        
            ofx = (ofx_perc * (output_lengths + tar_len_/2) - tar_len_/2) \
            .clamp(output_lengths*0, output_lengths - tar_len_ - 1).long()

            chunks_org = extract_chunks(mel_padded, ofx, mel_ids=None, chunk_len=tar_len_) # mel_padded: B F T
            chunks_gen = extract_chunks(pred_spect.transpose(1,2), ofx, mel_ids=None, chunk_len=tar_len_) # mel_out: B T F

            chunks_org_ = (chunks_org.unsqueeze(1) + 4.5) / 2.5
            chunks_gen_ = (chunks_gen.unsqueeze(1) + 4.5) / 2.5

        # generator        
        (loss, durs_loss, acc, acc_dist_1, acc_dist_3,
         pitch_loss, energy_loss, mel_loss, ctc_loss, bin_loss,) = model._metrics(
            pred_durs=pred_log_durs,
            pred_pitch=pred_pitch,
            pred_energy=pred_energy,
            true_durs=attn_hard_dur,
            true_text_len=input_lengths,
            true_pitch=pitch_padded[:,0],
            true_energy=energy_padded,
            true_spect=mel_padded,
            pred_spect=pred_spect,
            true_spect_len=output_lengths,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
        )
        
        meta = {
            'loss': loss,
            'durs_loss': durs_loss,
            'pitch_loss': torch.tensor(1.0).to(durs_loss.device) if pitch_loss is None else pitch_loss,
            'energy_loss': torch.tensor(1.0).to(durs_loss.device) if energy_loss is None else energy_loss,
            'mel_loss': mel_loss,
            'durs_acc': acc,
            'durs_acc_dist_3': acc_dist_3,
            'ctc_loss': torch.tensor(1.0).to(durs_loss.device) if ctc_loss is None else ctc_loss,
            'bin_loss': torch.tensor(1.0).to(durs_loss.device) if bin_loss is None else bin_loss,
        }
        
        if train_gan:
            
            # DISCRIMINATOR
            d_org, fmaps_org = critic(chunks_org_.requires_grad_(True))
            d_gen, _ = critic(chunks_gen_.detach())

            loss_d = 0.5*(d_org - 1).square().mean() + 0.5*(d_gen).square().mean()    

            critic.zero_grad()
            loss_d.backward()
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), 1000.)
            optimizer_d.step()
            
            meta['loss_d'] = loss_d.detach()
                    
            writer.add_scalar('train/gnorm_d', grad_norm_d, n_iter)

            # GENERATOR
            d_gen2, fmaps_gen = critic(chunks_gen_)
            loss_score = (d_gen2 - 1).square().mean()
            loss_fmatch = calc_feature_match_loss(fmaps_gen, fmaps_org)
     
            meta['score'] = loss_score.detach()
            meta['fmatch'] = loss_fmatch.detach()
                    
            loss += config.fmatch_loss*loss_fmatch            
            loss += config.score_loss*loss_score
            
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 20.)
        optimizer.step()

        if n_iter % 10 == 0:
            print(f"loss: {meta['loss'].item():.3f} mel: {mel_loss.item():.3f} gnorm: {grad_norm:.3f}")

        for k, v in meta.items():
            writer.add_scalar(f'train/{k}', v.item(), n_iter)
        writer.add_scalar('train/gnorm', grad_norm, n_iter)


        if n_iter % config.n_save_states_iter == 0:
            save_states(f'states.pth', model, critic,
                        optimizer, optimizer_d, n_iter, 
                        epoch, net_config, config)

        if n_iter % config.n_save_backup_iter == 0 and n_iter > 0:
            save_states(f'states_{n_iter}.pth', model, critic,
                        optimizer, optimizer_d, n_iter, 
                        epoch, net_config, config)

        n_iter += 1
    

save_states('states.pth', model, critic,
            optimizer, optimizer_d, n_iter,
            epoch, net_config, config)


# %%

idx = 0
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.imshow(pred_spect[idx].detach().cpu().t(), origin='lower', aspect='auto')
ax2.imshow(mel_padded[idx].cpu(), origin='lower', aspect='auto')

# %%

import sounddevice as sd
from vocos import Vocos
from utils.phonemizer import text_to_ids_en


sample_rate = [22050, 44100][0]

if sample_rate == 22050:
    vocos = Vocos.from_pretrained("BSC-LT/vocos-mel-22khz")
elif sample_rate == 44100:
    vocos = Vocos.from_pretrained("patriotyk/vocos-mel-hifigan-compat-44100khz")


# %%

text = """The basic Mixer-TTS contains pitch and duration predictors, with the latter being trained with an unsupervised TTS alignment framework."""

phonemes = text_to_ids_en(text)

# %%

model.eval()

phonemes_len = torch.LongTensor([len(phonemes)])

with torch.inference_mode():
    # (mel_out, dec_lens, dur_pred, pitch_pred, energy_pred) \
    mel_spec = model.infer(
             phonemes[None,:].to(device),
             phonemes_len.to(device),                    
             pace=1
    ).cpu()

wave = vocos.decode(mel_spec.transpose(1,2))
wave = wave / wave.abs().max()

sd.play(0.95*wave[0], sample_rate)

# %%

fig, ax = plt.subplots()
ax.imshow(mel_spec[0].T, origin='lower', aspect='auto', interpolation='none')

# %%
