import argparse
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import MixerTTSModel
from models.common.loss import PatchDiscriminator, calc_feature_match_loss, extract_chunks
from models.mixer_tts.modules.data_function import TTSCollate, batch_to_gpu
from utils import get_config
from utils.lj_dataset import DynBatchDataset, LJDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/synthetic-ft-80.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=50)
    return parser.parse_args()


def split_indices(n_items, eval_fraction, seed):
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    n_eval = max(1, int(round(n_items * eval_fraction)))
    return indices[n_eval:], indices[:n_eval]


def read_pitch_stats(pitch_dir):
    stats_path = Path(pitch_dir) / "mean_std.txt"
    values = {}
    for line in stats_path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key.strip()] = float(value.strip())
    return values["mean"], values["std"]


def save_checkpoint(path, model, critic, optimizer, optimizer_d, n_iter, epoch, net_config, config, best_eval_loss):
    payload = {
        "model": model.state_dict(),
        "model_d": critic.state_dict(),
        "optim": optimizer.state_dict(),
        "optim_d": optimizer_d.state_dict(),
        "epoch": epoch,
        "iter": n_iter,
        "net_config": net_config,
        "best_eval_loss": best_eval_loss,
    }
    torch.save(payload, path)


def load_model(config, device):
    ckpt = torch.load(config.restore_model, map_location=device, weights_only=False)
    net_config = dict(ckpt["net_config"])
    model = MixerTTSModel(**net_config).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model, net_config


def forward_losses(model, batch):
    (
        text_padded,
        input_lengths,
        mel_padded,
        output_lengths,
        pitch_padded,
        energy_padded,
        speaker,
        emotion,
        attn_prior_padded,
        _,
    ), _, _ = batch_to_gpu(batch)

    (
        pred_spect,
        _,
        pred_log_durs,
        pred_pitch,
        pred_energy,
        attn_soft,
        attn_logprob,
        attn_hard,
        attn_hard_dur,
    ) = model(
        text=text_padded,
        text_len=input_lengths,
        pitch=pitch_padded[:, 0],
        energy=energy_padded,
        spect=mel_padded,
        spect_len=output_lengths,
        attn_prior=attn_prior_padded,
        lm_tokens=None,
        speaker=speaker,
        emotion=emotion,
    )

    (
        loss,
        durs_loss,
        acc,
        _,
        acc_dist_3,
        pitch_loss,
        energy_loss,
        mel_loss,
        ctc_loss,
        bin_loss,
    ) = model._metrics(
        pred_durs=pred_log_durs,
        pred_pitch=pred_pitch,
        pred_energy=pred_energy,
        true_durs=attn_hard_dur,
        true_text_len=input_lengths,
        true_pitch=pitch_padded[:, 0],
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
        "loss": loss,
        "durs_loss": durs_loss,
        "pitch_loss": torch.zeros_like(loss) if pitch_loss is None else pitch_loss,
        "energy_loss": torch.zeros_like(loss) if energy_loss is None else energy_loss,
        "mel_loss": mel_loss,
        "durs_acc": acc,
        "durs_acc_dist_3": acc_dist_3,
        "ctc_loss": torch.zeros_like(loss) if ctc_loss is None else ctc_loss,
        "bin_loss": torch.zeros_like(loss) if bin_loss is None else bin_loss,
    }
    return loss, meta, pred_spect, mel_padded, output_lengths


@torch.inference_mode()
def evaluate(model, eval_loader, max_batches=50):
    was_training = model.training
    model.eval()
    losses = []
    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break
        loss, _, _, _, _ = forward_losses(model, batch)
        losses.append(loss.detach())
    if was_training:
        model.train()
    if not losses:
        return math.inf
    return torch.stack(losses).mean().item()


def main():
    args = parse_args()
    config = get_config(args.config)
    device = "cuda:0" if torch.cuda.is_available() and config.use_cuda_if_available else "cpu"

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    f0_mean, f0_std = read_pitch_stats(config.pitch_dir)

    train_dataset_dyn = DynBatchDataset(
        audio_dir=config.train_audio_dir,
        textfile_path=config.train_labels,
        pitch_dir=config.pitch_dir,
        rms_db=None,
        f_cutoff=None,
        f0_mean=f0_mean,
        f0_std=f0_std,
    )
    train_indices, eval_indices = split_indices(
        len(train_dataset_dyn.data),
        getattr(config, "eval_fraction", 0.02),
        getattr(config, "split_seed", 1337),
    )
    train_dataset_dyn.data = [train_dataset_dyn.data[i] for i in train_indices]
    train_dataset_dyn.shuffle()

    eval_dataset = LJDataset(
        audio_dir=config.train_audio_dir,
        textfile_path=config.train_labels,
        pitch_dir=config.pitch_dir,
        rms_db=None,
        f_cutoff=None,
        f0_mean=f0_mean,
        f0_std=f0_std,
    )
    eval_dataset = Subset(eval_dataset, eval_indices)

    collate_fn = TTSCollate()
    train_loader = DataLoader(
        train_dataset_dyn,
        batch_size=1,
        collate_fn=lambda x: collate_fn(x[0]),
        shuffle=True,
        drop_last=True,
        num_workers=getattr(config, "num_workers", 0),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=8,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=getattr(config, "num_workers", 0),
    )

    model, net_config = load_model(config, device)
    model.add_bin_loss = True
    model.bin_loss_scale = 1.0

    train_gan = getattr(config, "train_gan", True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.0, 0.99) if train_gan else (0.9, 0.98),
    )
    critic = PatchDiscriminator(1, 32).to(device)
    optimizer_d = torch.optim.AdamW(
        critic.parameters(),
        lr=1e-4,
        betas=(0.0, 0.99),
        weight_decay=config.weight_decay,
    )

    writer = SummaryWriter(config.log_dir)
    best_eval_loss = math.inf
    n_iter = 0
    epoch = 0
    tar_len = 128
    print(f"device={device}")
    print(f"train_items={len(train_dataset_dyn.data)} eval_items={len(eval_dataset)}")
    print(f"restore_model={config.restore_model}")
    print(f"checkpoint_dir={config.checkpoint_dir}")
    print(f"f0_mean={f0_mean:.6f} f0_std={f0_std:.6f}")

    model.train()
    critic.train()
    while True:
        train_dataset_dyn.shuffle()
        progress = tqdm(train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in progress:
            loss, meta, pred_spect, mel_padded, output_lengths = forward_losses(model, batch)

            if train_gan:
                tar_len_ = min(output_lengths.min(), tar_len)
                ofx_perc = torch.rand(output_lengths.size()).to(device)
                ofx = (ofx_perc * (output_lengths + tar_len_ / 2) - tar_len_ / 2).clamp(
                    output_lengths * 0, output_lengths - tar_len_ - 1
                ).long()
                chunks_org = extract_chunks(mel_padded, ofx, mel_ids=None, chunk_len=tar_len_)
                chunks_gen = extract_chunks(pred_spect.transpose(1, 2), ofx, mel_ids=None, chunk_len=tar_len_)
                chunks_org_ = (chunks_org.unsqueeze(1) + 4.5) / 2.5
                chunks_gen_ = (chunks_gen.unsqueeze(1) + 4.5) / 2.5

                d_org, fmaps_org = critic(chunks_org_.requires_grad_(True))
                d_gen, _ = critic(chunks_gen_.detach())
                loss_d = 0.5 * (d_org - 1).square().mean() + 0.5 * d_gen.square().mean()

                optimizer_d.zero_grad()
                loss_d.backward()
                grad_norm_d = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1000.0)
                optimizer_d.step()

                d_gen2, fmaps_gen = critic(chunks_gen_)
                loss_score = (d_gen2 - 1).square().mean()
                loss_fmatch = calc_feature_match_loss(fmaps_gen, fmaps_org)
                loss = loss + config.fmatch_loss * loss_fmatch + config.score_loss * loss_score
                meta["loss_d"] = loss_d.detach()
                meta["score"] = loss_score.detach()
                meta["fmatch"] = loss_fmatch.detach()
                writer.add_scalar("train/gnorm_d", grad_norm_d, n_iter)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            optimizer.step()

            if n_iter % config.print_interval == 0:
                progress.write(
                    f"iter={n_iter} epoch={epoch} "
                    f"loss={meta['loss'].item():.4f} mel={meta['mel_loss'].item():.4f} "
                    f"gnorm={float(grad_norm):.4f}"
                )
            progress.set_postfix(
                iter=n_iter,
                loss=f"{meta['loss'].item():.3f}",
                mel=f"{meta['mel_loss'].item():.3f}",
            )
            for k, v in meta.items():
                writer.add_scalar(f"train/{k}", v.item(), n_iter)
            writer.add_scalar("train/gnorm", grad_norm, n_iter)

            if n_iter % config.save_interval == 0:
                save_checkpoint(
                    os.path.join(config.checkpoint_dir, "last.pth"),
                    model,
                    critic,
                    optimizer,
                    optimizer_d,
                    n_iter,
                    epoch,
                    net_config,
                    config,
                    best_eval_loss,
                )

            if n_iter % config.eval_interval == 0:
                eval_loss = evaluate(model, eval_loader, max_batches=args.eval_max_batches)
                writer.add_scalar("eval/loss", eval_loss, n_iter)
                progress.write(f"iter={n_iter} eval_loss={eval_loss:.4f} best={best_eval_loss:.4f}")
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(
                        os.path.join(config.checkpoint_dir, "best.pth"),
                        model,
                        critic,
                        optimizer,
                        optimizer_d,
                        n_iter,
                        epoch,
                        net_config,
                        config,
                        best_eval_loss,
                    )
                    progress.write(f"saved new best: {best_eval_loss:.4f}")

            n_iter += 1
            if args.max_steps is not None and n_iter >= args.max_steps:
                print(f"reached max_steps={args.max_steps}; exiting", flush=True)
                return
        epoch += 1


if __name__ == "__main__":
    main()
