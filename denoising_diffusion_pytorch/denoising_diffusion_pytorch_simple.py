import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import einops
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad
from torch.func import grad as f_grad
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA
import numpy as np
from tqdm.auto import tqdm
from denoising_diffusion_pytorch.version import __version__
import wandb

# constants

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])

# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    # num = self.num_samples
    # divisor = self.batch_size
    groups = num // divisor  # 1
    remainder = num % divisor  # 0
    arr = [divisor] * groups  #
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# data


class Dataset1D(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()


# small helper modules


class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds


class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


class Block(Module):
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c n -> b (h c) n", h=self.heads)
        return self.to_out(out)


class Attention(Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b (h d) n")
        return self.to_out(out)


# model


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        embed = x.unsqueeze(-2)
        embed = repeat(embed, "... j n -> ... (k j) n", k=2 * self.num_freqs)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        embed = rearrange(embed, "... j n -> ... (j n)")

        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        return embed


class ModelBasic(nn.Module):
    def __init__(self, data_dim=2):
        super().__init__()

        self.data_dim = data_dim
        self.pos_enc = PositionalEncoding(freq_factor=1.5, d_in=data_dim + 1)
        in_dim = self.pos_enc.d_out
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256 + in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.out = nn.Linear(2, 2)

        self.channels = self.data_dim

        self.self_condition = False
        self.predict_latent = True

    def get_feats(self, inp):
        inp = self.pos_enc(inp)
        hidden = self.block1(inp)
        latent = self.block2(torch.cat([hidden, inp], dim=1))
        latent = self.out(latent)
        return latent

    def forward(self, model_input, t, x_self_cond=None):

        concat_tensor = torch.cat([model_input, t.unsqueeze(1)], dim=1)
        out = self.get_feats(concat_tensor)

        return out


class ResnetDiffusionModel(nn.Module):
    """Resnet score model with embedding for each scale after each linear layer."""

    def __init__(
        self, n_steps, n_layers, x_dim, h_dim, emb_dim, widen=2, emb_type="learned"
    ):
        assert emb_type in ("learned", "sinusoidal")
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.x_dim = x_dim
        self.channels = self.x_dim
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.widen = widen
        self.emb_type = emb_type

        # Embedding layer
        if self.emb_type == "learned":
            self.embedding = nn.Embedding(n_steps, emb_dim)
        else:
            self.embedding = self.timestep_embedding

        # Initial linear layer
        self.initial_linear = nn.Linear(x_dim, h_dim)

        # ResNet layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "layer_h": nn.Linear(h_dim, h_dim * widen),
                        "layer_emb": nn.Linear(emb_dim, h_dim * widen),
                        "layer_int": nn.Linear(h_dim * widen, h_dim * widen),
                        "layer_out": nn.Linear(h_dim * widen, h_dim, bias=False),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        # Final linear layer
        self.final_linear = nn.Linear(h_dim, x_dim, bias=False)

        # LayerNorm
        self.layer_norm = nn.LayerNorm(h_dim)

    def timestep_embedding(self, t, emb_dim):
        """Generates sinusoidal embeddings for the timesteps."""
        half_dim = emb_dim // 2
        emb = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32)
            * torch.log(torch.tensor(10000.0))
            / half_dim
        )
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x, t, x_self_cond=None):
        x = x if x.ndim == 2 else x.view(x.size(0), -1)
        t = t if t.ndim == 1 else t.view(-1)

        assert x.size(1) == self.x_dim, "Input x must have dimension x_dim"
        assert t.size(0) == x.size(0), "Batch size of t must match x"

        # Get embedding
        if self.emb_type == "learned":
            emb = self.embedding(t)
        else:
            emb = self.timestep_embedding(t, self.emb_dim)

        # Initial layer
        x = self.initial_linear(x)

        # ResNet layers
        for layer in self.layers:
            h = self.layer_norm(x)
            h = F.silu(h)
            h = layer["layer_h"](h)
            h = h + layer["layer_emb"](emb)
            h = F.silu(h)
            h = layer["layer_int"](h)
            h = F.silu(h)
            h = layer["layer_out"](h)
            x = x + h

        # Final layer
        x = self.final_linear(x)
        return x


# we should try out a bunch of different parametrizations here
class EBMDiffusionModel(nn.Module):

    def __init__(self, net: ResnetDiffusionModel):
        super().__init__()
        self.net = net
        self.channels = self.net.channels

    def neg_logp_unnorm(self, x, t, x_cond):
        score = self.net(x, t, x_cond)
        return ((score - x) ** 2).sum(dim=-1)

    def forward(self, x, t, x_cond):
        with torch.enable_grad():

            def neg_logp_unnorm_fn(_x):
                return self.neg_logp_unnorm(_x, t, x_cond).sum()

            # Compute gradient with respect to x
            if not x.requires_grad:
                x.requires_grad_(True)

            neg_logp = neg_logp_unnorm_fn(x)
            grad_output = grad(neg_logp, x, create_graph=True, retain_graph=True)[0]

        return grad_output


class ConditionalResnetDiffusionModel(nn.Module):
    """Resnet score model with embedding for each scale after each linear layer."""

    def __init__(
        self,
        n_steps,
        n_layers,
        x_dim,
        h_dim,
        emb_dim,
        cond_y_dim,
        num_conds=None,
        widen=2,
        emb_type="learned",
    ):
        assert emb_type in ("learned", "sinusoidal")
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.x_dim = x_dim
        self.cond_y_dim = cond_y_dim
        self.channels = self.x_dim
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.widen = widen
        self.emb_type = emb_type
        self.cond_is_index = num_conds is not None
        self.num_conds = num_conds

        # Embedding layer
        if self.emb_type == "learned":
            self.embedding = nn.Embedding(n_steps, emb_dim)
        else:
            self.embedding = self.timestep_embedding

        if self.cond_is_index:
            self.cond_pre_embedding = nn.Embedding(num_conds, cond_y_dim)

        # Initial linear layer
        self.initial_linear = nn.Linear(x_dim, h_dim)
        self.cond_channels = h_dim * 2

        # ResNet layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "layer_h": nn.Linear(h_dim, h_dim * widen),
                        "layer_emb": nn.Linear(emb_dim, h_dim * widen),
                        "layer_int": nn.Linear(h_dim * widen, h_dim * widen),
                        "layer_out": nn.Linear(h_dim * widen, h_dim, bias=False),
                        "cond_encoder": nn.Sequential(
                            nn.Mish(),
                            nn.Linear(cond_y_dim, self.cond_channels),
                            Rearrange("b c -> b c 1"),
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        # we will be using FiLM conditioning

        # Final linear layer
        self.final_linear = nn.Linear(h_dim, x_dim, bias=False)

        # LayerNorm
        self.layer_norm = nn.LayerNorm(h_dim)

    def timestep_embedding(self, t, emb_dim):
        """Generates sinusoidal embeddings for the timesteps."""
        half_dim = emb_dim // 2
        emb = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32)
            * torch.log(torch.tensor(10000.0))
            / half_dim
        )
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x, cond, t, x_self_cond=None):
        x = x if x.ndim == 2 else x.view(x.size(0), -1)
        t = t if t.ndim == 1 else t.view(-1)

        assert x.size(1) == self.x_dim, "Input x must have dimension x_dim"
        assert t.size(0) == x.size(0), "Batch size of t must match x"
        assert cond.size(0) == x.size(0), "Batch size of cond must match x"
        if not self.cond_is_index:
            assert (
                cond.size(1) == self.cond_y_dim
            ), "Input cond must have dimension cond_y_dim"

        # Get embedding
        if self.emb_type == "learned":
            emb = self.embedding(t)
        else:
            emb = self.timestep_embedding(t, self.emb_dim)

        # cond is index so get embedding
        if self.cond_is_index:
            assert cond.shape[-1] == 1, "conditional input must be index"
            assert cond.dtype == torch.int64, "conditional input must be LongTensor"
            cond = self.cond_pre_embedding(cond.squeeze(-1))

        # Initial layer
        x = self.initial_linear(x)

        # ResNet layers
        for layer in self.layers:
            h = self.layer_norm(x)
            h = F.silu(h)
            h = layer["layer_h"](h)
            h = h + layer["layer_emb"](emb)
            h = F.silu(h)
            h = layer["layer_int"](h)
            h = F.silu(h)
            h = layer["layer_out"](h)
            x = x + h
            film_params = layer["cond_encoder"](cond)
            film_params = film_params.reshape(film_params.shape[0], 2, self.h_dim)
            scale = film_params[:, 0, ...]
            bias = film_params[:, 1, ...]
            x = x * scale + bias

        # Final layer
        x = self.final_linear(x)
        return x


# we should try out a bunch of different parametrizations here
class EBMConditionalDiffusionModel(nn.Module):

    def __init__(self, net: ConditionalResnetDiffusionModel):
        super().__init__()
        self.net = net
        self.channels = self.net.channels
        self.num_conds = self.net.num_conds

    def neg_logp_unnorm(self, x, x_cond, t, self_x_cond):
        score = self.net(x, x_cond, t, self_x_cond)
        return ((score - x) ** 2).sum(dim=-1)

    def forward(self, x, x_cond, t, self_x_cond):
        with torch.enable_grad():

            def neg_logp_unnorm_fn(_x):
                return self.neg_logp_unnorm(_x, x_cond, t, self_x_cond).sum()

            # Compute gradient with respect to x
            if not x.requires_grad:
                x.requires_grad_(True)

            neg_logp = neg_logp_unnorm_fn(x)
            grad_output = grad(neg_logp, x, create_graph=True, retain_graph=True)[0]

        return grad_output


class FunctionalEBMConditionalDiffusionModel(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.channels = self.net.channels
        self.num_conds = self.net.num_conds

    def neg_logp_unnorm(self, x, x_cond, t, self_x_cond):
        # Calculate the unnormalized negative log probability
        score = self.net(x, x_cond, t, self_x_cond)
        return ((score - x) ** 2).sum(dim=-1)

    def forward(self, x, x_cond, t, self_x_cond):
        # Define a function that computes the sum of the negative log probabilities
        def neg_logp_unnorm_fn(_x):
            return self.neg_logp_unnorm(_x, x_cond, t, self_x_cond).sum()

        # Use torch.func.grad to compute the gradient of the neg_logp_unnorm function
        gradient_fn = f_grad(neg_logp_unnorm_fn)
        grad_output = gradient_fn(x)

        return grad_output


class ScoreFactoredDiffusionModel(nn.Module):
    """Resnet score model with embedding for each scale after each linear layer."""

    def __init__(
        self,
        n_steps,
        n_layers,
        x_dim,
        h_dim,
        emb_dim,
        cond_y_dim,
        num_conds=None,
        widen=2,
        normalize=True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.x_dim = x_dim
        self.cond_y_dim = cond_y_dim
        self.channels = self.x_dim
        h_dim = (
            x_dim * h_dim
        )  # unique feature of this model because of how it is factorized
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.widen = widen
        self.cond_is_index = num_conds is not None
        self.num_conds = num_conds
        self.normalize = normalize
        self.embedding = nn.Embedding(n_steps, emb_dim)

        # Layers for variable being generated
        self.initial_linear = nn.Linear(x_dim, h_dim)
        self.layer_norm = nn.LayerNorm(h_dim)

        self.x_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "layer_h": nn.Linear(h_dim, h_dim * widen),
                        "layer_emb": nn.Linear(emb_dim, h_dim * widen),
                        "layer_int": nn.Linear(h_dim * widen, h_dim * widen),
                        "layer_out": nn.Linear(h_dim * widen, h_dim, bias=False),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        if self.cond_is_index:
            self.cond_pre_embedding = nn.Embedding(num_conds, cond_y_dim)

        self.initial_layer_cond = nn.Linear(cond_y_dim, h_dim)
        self.cond_layer_norm = nn.LayerNorm(h_dim)

        self.cond_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "layer_h": nn.Linear(h_dim, h_dim * widen),
                        "layer_emb": nn.Linear(emb_dim, h_dim * widen),
                        "layer_int": nn.Linear(h_dim * widen, h_dim * widen),
                        "layer_out": nn.Linear(h_dim * widen, h_dim, bias=False),
                    }
                )
                for _ in range(n_layers)
            ]
        )

    def timestep_embedding(self, t, emb_dim):
        """Generates sinusoidal embeddings for the timesteps."""
        half_dim = emb_dim // 2
        emb = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32)
            * torch.log(torch.tensor(10000.0))
            / half_dim
        )
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x, cond, t, x_self_cond=None):
        x = x if x.ndim == 2 else x.view(x.size(0), -1)
        t = t if t.ndim == 1 else t.view(-1)

        assert x.size(1) == self.x_dim, "Input x must have dimension x_dim"
        assert t.size(0) == x.size(0), "Batch size of t must match x"
        assert cond.size(0) == x.size(0), "Batch size of cond must match x"
        if not self.cond_is_index:
            assert (
                cond.size(1) == self.cond_y_dim
            ), "Input cond must have dimension cond_y_dim"

        # Get embedding

        time_emb = self.embedding(t)

        # Initial layer
        x = self.initial_linear(x)
        # ResNet layers for x
        for layer in self.x_layers:
            h = self.layer_norm(x)
            h = F.silu(h)
            h = layer["layer_h"](h)
            h = h + layer["layer_emb"](time_emb)
            h = F.silu(h)
            h = layer["layer_int"](h)
            h = F.silu(h)
            h = layer["layer_out"](h)
            x = x + h

        x = einops.rearrange(x, "b (x h) -> b x h", x=self.x_dim)

        # cond is index so get embedding
        if self.cond_is_index:
            assert cond.shape[-1] == 1, "conditional input must be index"
            assert cond.dtype == torch.int64, "conditional input must be LongTensor"
            cond = self.cond_pre_embedding(cond.squeeze(-1))

        # Initial layer for cond
        cond = self.initial_layer_cond(cond)
        for layer in self.cond_layers:
            h = self.cond_layer_norm(cond)
            h = F.silu(h)
            h = layer["layer_h"](h)
            h = h + layer["layer_emb"](time_emb)
            h = F.silu(h)
            h = layer["layer_int"](h)
            h = F.silu(h)
            h = layer["layer_out"](h)
            cond = cond + h

        cond = einops.rearrange(cond, "b (c h) -> b c h", c=self.x_dim)
        if self.normalize:
            x_feats = x / x.norm(dim=-1, keepdim=True)
            cond_feats = cond / cond.norm(dim=-1, keepdim=True)
        else:
            x_feats = x
            cond_feats = cond

        score = (x_feats * cond_feats).sum(dim=-1)
        return score


class EBMFactoredDiffusionModel(nn.Module):
    """Resnet score model with embedding for each scale after each linear layer."""

    def __init__(
        self,
        n_steps,
        n_layers,
        x_dim,
        h_dim,
        emb_dim,
        cond_y_dim,
        num_conds=None,
        widen=2,
        normalize=True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.x_dim = x_dim
        self.cond_y_dim = cond_y_dim
        self.channels = self.x_dim
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.widen = widen
        self.cond_is_index = num_conds is not None
        self.num_conds = num_conds
        self.normalize = normalize

        self.embedding = nn.Embedding(n_steps, emb_dim)

        # Layers for variable being generated
        self.initial_linear = nn.Linear(x_dim, h_dim)
        self.layer_norm = nn.LayerNorm(h_dim)

        self.x_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "layer_h": nn.Linear(h_dim, h_dim * widen),
                        "layer_emb": nn.Linear(emb_dim, h_dim * widen),
                        "layer_int": nn.Linear(h_dim * widen, h_dim * widen),
                        "layer_out": nn.Linear(h_dim * widen, h_dim, bias=False),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        if self.cond_is_index:
            self.cond_pre_embedding = nn.Embedding(num_conds, cond_y_dim)

        self.initial_layer_cond = nn.Linear(cond_y_dim, h_dim)
        self.cond_layer_norm = nn.LayerNorm(h_dim)

        self.cond_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "layer_h": nn.Linear(h_dim, h_dim * widen),
                        "layer_emb": nn.Linear(emb_dim, h_dim * widen),
                        "layer_int": nn.Linear(h_dim * widen, h_dim * widen),
                        "layer_out": nn.Linear(h_dim * widen, h_dim, bias=False),
                    }
                )
                for _ in range(n_layers)
            ]
        )

    def timestep_embedding(self, t, emb_dim):
        """Generates sinusoidal embeddings for the timesteps."""
        half_dim = emb_dim // 2
        emb = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32)
            * torch.log(torch.tensor(10000.0))
            / half_dim
        )
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def neg_logp_unnorm(self, x, cond, t, x_self_cond=None):
        x = x if x.ndim == 2 else x.view(x.size(0), -1)
        t = t if t.ndim == 1 else t.view(-1)

        assert x.size(1) == self.x_dim, "Input x must have dimension x_dim"
        assert t.size(0) == x.size(0), "Batch size of t must match x"
        assert cond.size(0) == x.size(0), "Batch size of cond must match x"
        if not self.cond_is_index:
            assert (
                cond.size(1) == self.cond_y_dim
            ), "Input cond must have dimension cond_y_dim"

        # Get embedding

        time_emb = self.embedding(t)

        # Initial layer
        x = self.initial_linear(x)
        # ResNet layers for x
        for layer in self.x_layers:
            h = self.layer_norm(x)
            h = F.silu(h)
            h = layer["layer_h"](h)
            h = h + layer["layer_emb"](time_emb)
            h = F.silu(h)
            h = layer["layer_int"](h)
            h = F.silu(h)
            h = layer["layer_out"](h)
            x = x + h

        # cond is index so get embedding
        if self.cond_is_index:
            assert cond.shape[-1] == 1, "conditional input must be index"
            assert cond.dtype == torch.int64, "conditional input must be LongTensor"
            cond = self.cond_pre_embedding(cond.squeeze(-1))

        # Initial layer for cond
        cond = self.initial_layer_cond(cond)
        for layer in self.cond_layers:
            h = self.cond_layer_norm(cond)
            h = F.silu(h)
            h = layer["layer_h"](h)
            h = h + layer["layer_emb"](time_emb)
            h = F.silu(h)
            h = layer["layer_int"](h)
            h = F.silu(h)
            h = layer["layer_out"](h)
            cond = cond + h

        if self.normalize:
            x_feats = x / x.norm(dim=-1, keepdim=True)
            cond_feats = cond / cond.norm(dim=-1, keepdim=True)
        else:
            x_feats = x
            cond_feats = cond

        dot_product = (x_feats * cond_feats).sum(dim=-1)

        return dot_product

    def forward(self, x, x_cond, t, self_x_cond):
        with torch.enable_grad():

            def neg_logp_unnorm_fn(_x):
                return self.neg_logp_unnorm(_x, x_cond, t, self_x_cond).sum()

            # Compute gradient with respect to x
            if not x.requires_grad:
                x.requires_grad_(True)

            neg_logp = neg_logp_unnorm_fn(x)
            grad_output = grad(neg_logp, x, create_graph=True, retain_graph=True)[0]

        return grad_output


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusionSimple(Module):
    def __init__(
        self,
        model,
        *,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_noise",
        beta_schedule="cosine",
        ddim_sampling_eta=0.0,
        auto_normalize=False,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = False

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == "pred_noise":
            loss_weight = torch.ones_like(snr)
        elif objective == "pred_x0":
            loss_weight = snr
        elif objective == "pred_v":
            loss_weight = snr / (snr + 1)

        register_buffer("loss_weight", loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False
    ):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device
        # TODO: "img" is confusing as there are other things to do here
        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, self_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        channels = self.channels
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn((batch_size, channels))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        b, c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, given_tensor, *args, **kwargs):
        (
            b,
            c,
            device,
        ) = (
            *given_tensor.shape,
            given_tensor.device,
        )
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        given_tensor = self.normalize(given_tensor)
        return self.p_losses(given_tensor, t, *args, **kwargs)


class ConditionalGaussianDiffusionSimple(Module):
    def __init__(
        self,
        model,
        *,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_noise",
        beta_schedule="cosine",
        ddim_sampling_eta=0.0,
        auto_normalize=False,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = False

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == "pred_noise":
            loss_weight = torch.ones_like(snr)
        elif objective == "pred_x0":
            loss_weight = snr
        elif objective == "pred_v":
            loss_weight = snr / (snr + 1)

        register_buffer("loss_weight", loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x,
        x_cond,
        t,
        x_self_cond=None,
        clip_x_start=False,
        rederive_pred_noise=False,
    ):
        model_output = self.model(x, x_cond, t, x_self_cond)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self, x, sample_x_cond, t, x_self_cond=None, clip_denoised=True
    ):
        preds = self.model_predictions(x, sample_x_cond, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, sample_x_cond, t: int, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            sample_x_cond=sample_x_cond,
            t=batched_times,
            x_self_cond=x_self_cond,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, sample_x_cond):
        batch, device = shape[0], self.betas.device
        # TODO: "img" is confusing as there are other things to do here
        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, sample_x_cond, t, self_cond)

        img = self.unnormalize(img)
        return img, sample_x_cond

    # @torch.no_grad()
    # def ddim_sample(self, shape, clip_denoised=True):
    #     batch, device, total_timesteps, sampling_timesteps, eta, objective = (
    #         shape[0],
    #         self.betas.device,
    #         self.num_timesteps,
    #         self.sampling_timesteps,
    #         self.ddim_sampling_eta,
    #         self.objective,
    #     )

    #     times = torch.linspace(
    #         -1, total_timesteps - 1, steps=sampling_timesteps + 1
    #     )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(
    #         zip(times[:-1], times[1:])
    #     )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    #     img = torch.randn(shape, device=device)

    #     x_start = None

    #     for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
    #         time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
    #         self_cond = x_start if self.self_condition else None
    #         pred_noise, x_start, *_ = self.model_predictions(
    #             img, time_cond, self_cond, clip_x_start=clip_denoised
    #         )

    #         if time_next < 0:
    #             img = x_start
    #             continue

    #         alpha = self.alphas_cumprod[time]
    #         alpha_next = self.alphas_cumprod[time_next]

    #         sigma = (
    #             eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         )
    #         c = (1 - alpha_next - sigma**2).sqrt()

    #         noise = torch.randn_like(img)

    #         img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

    #     img = self.unnormalize(img)
    #     return img

    @torch.no_grad()
    def sample(self, sample_x_cond, batch_size=16):

        channels = self.channels
        if self.is_ddim_sampling:
            raise NotImplementedError(
                "DDIM sampling not implemented for conditional diffusion"
            )
        sample_fn = self.p_sample_loop
        return sample_fn((batch_size, channels), sample_x_cond)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, x_cond, t, noise=None):
        b, c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, x_cond, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, x_cond, t, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, given_batch, *args, **kwargs):

        given_tensor = given_batch["tensor"]
        (
            b,
            c,
            device,
        ) = (
            *given_tensor.shape,
            given_tensor.device,
        )

        given_cond = given_batch["cond"]
        (
            b_cond,
            c_cond,
            device,
        ) = (
            *given_cond.shape,
            given_cond.device,
        )

        assert b == b_cond, "batch size of tensor and cond must be the same"

        if given_cond.dtype == torch.int64:
            assert given_cond.shape[-1] == 1, "cond must be a tensor of shape (b, 1)"

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        given_tensor = self.normalize(given_tensor)

        # check if cond is a FloatTensor or LongTensor
        if not given_cond.dtype == torch.int64:
            raise NotImplementedError(
                "cond must be a for now LongTensor but modify normalization here if that changes"
            )

        return self.p_losses(given_tensor, given_cond, t, *args, **kwargs)


class TrainerSimple(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusionSimple,
        dataset: Dataset,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        n_epochs=100,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=40000,
        num_samples=100,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
        conditional_sampling=False,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else "no",
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(
            num_samples
        ), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm
        self.conditional_sampling = conditional_sampling
        self.n_epochs = n_epochs

        # dataset and dataloader

        dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=10,
        )

        self.dl = dl
        # dl = self.accelerator.prepare(dl)
        # self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        # if self.accelerator.is_main_process:
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        # self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler)
                else None
            ),
            "version": __version__,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(self.results_folder / f"model-{milestone}.pt"),
            map_location=device,
            weights_only=True,
        )

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train_without_accelerate(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.train()
        min_loss = 1e9
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for idx, data in tqdm(enumerate(self.dl)):
                # check if data is a single tensor or a dict
                if isinstance(data, dict):
                    data = {k: v.to(device) for k, v in data.items()}
                else:
                    data = data.to(device)

                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):

                    loss = self.model(data)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                    loss.backward()

                epoch_loss += total_loss

                wandb.log({"loss": total_loss})

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                self.ema.update()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema.ema_model.eval()

                    with torch.no_grad():
                        milestone = self.step // self.save_and_sample_every
                        batches = num_to_groups(self.num_samples, self.batch_size)
                        if self.conditional_sampling == "discrete":
                            num_classes = self.model.model.num_conds
                            all_generated_list = list(
                                map(
                                    lambda n: self.ema.ema_model.sample(
                                        batch_size=n,
                                        sample_x_cond=torch.randint(
                                            0, num_classes, (n, 1)
                                        )
                                        .to(torch.int64)
                                        .to(device),
                                    ),
                                    batches,
                                )
                            )

                            all_samples_list = [
                                gen_samples[0] for gen_samples in all_generated_list
                            ]
                            all_conditions_list = [
                                gen_samples[1] for gen_samples in all_generated_list
                            ]

                        else:
                            all_samples_list = list(
                                map(
                                    lambda n: self.ema.ema_model.sample(batch_size=n),
                                    batches,
                                )
                            )
                            all_conditions_list = [
                                torch.zeros_like(each)[:, 0:1].to(torch.int64)
                                for each in all_samples_list
                            ]

                    all_samples = torch.cat(all_samples_list, dim=0)
                    all_conditions = torch.cat(all_conditions_list, dim=0)

                    assert self.channels >= 2
                    # plot all samples on a 2D grid
                    x = all_samples[:, 0].cpu().numpy()
                    y = all_samples[:, 1].cpu().numpy()
                    all_conditions = list(all_conditions[:, 0].cpu().numpy())
                    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
                    import matplotlib.pyplot as plt
                    from io import BytesIO
                    from PIL import Image

                    plt.figure(figsize=(6, 4))
                    plt.xlim(
                        -1, 1
                    )  # Set x-axis limits to include negative and positive values
                    plt.ylim(
                        -1, 1
                    )  # Set y-axis limits to include negative and positive values
                    plt.scatter(
                        x,
                        y,
                        c=[colors[each] for each in all_conditions],
                        label="Samples",
                    )
                    plt.grid(True)

                    # Save the plot to a BytesIO object
                    buffer = BytesIO()
                    plt.savefig(buffer, format="png")
                    buffer.seek(0)

                    # Log the plot to wandb
                    wandb.log({"samples": wandb.Image(Image.open(buffer))})

                    # torch.save(
                    #     all_samples,
                    #     str(self.results_folder / f"sample-{milestone}.png"),
                    # )
                    # self.save(milestone)

            epoch_loss /= len(self.dl)
            print(f"Epoch {epoch}: Loss {epoch_loss}")
            min_loss = min(min_loss, epoch_loss)

        return min_loss
