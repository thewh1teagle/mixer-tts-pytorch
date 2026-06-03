from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm


LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if module.__class__.__name__.find("Conv") != -1:
        module.weight.data.normal_(mean, std)


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int, int] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
                for _ in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int] = (1, 3)) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            remove_weight_norm(layer)


class HiFiGANGenerator(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(config.num_mels, config.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if config.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            in_channels = config.upsample_initial_channel // (2**i)
            out_channels = config.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        in_channels,
                        out_channels,
                        kernel,
                        rate,
                        padding=(kernel - rate) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(resblock(channels, kernel, tuple(dilation)))

        self.conv_post = weight_norm(Conv1d(channels, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                block_out = self.resblocks[i * self.num_kernels + j](x)
                xs = block_out if xs is None else xs + block_out
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def load_hifigan(config_path: str | Path, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> HiFiGANGenerator:
    config = SimpleNamespace(**json.loads(Path(config_path).read_text(encoding="utf-8")))
    model = HiFiGANGenerator(config).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state["generator"] if isinstance(state, dict) and "generator" in state else state
    model.load_state_dict(state_dict)
    model.remove_weight_norm()
    model.eval()
    return model
