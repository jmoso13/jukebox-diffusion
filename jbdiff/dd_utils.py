from copy import deepcopy
import math

import os, signal, sys
import gc

from diffusion import sampling
import torch
from torch import optim, nn
from torch.nn import functional as F

from audio_diffusion.models import DiffusionAttnUnet1D
import numpy as np

from audio_diffusion.utils import Stereo, PadCrop
from glob import glob


class DiffusionUncond(nn.Module):
    def __init__(self, global_args):
        super().__init__()

        self.diffusion = DiffusionAttnUnet1D(global_args, n_attn_layers = 4)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)


def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)


def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


def resample(model_fn, audio, steps=100, sampler_type="v-ddim", noise_level = 1.0):
  #Noise the input
  if sampler_type.startswith("v-"):
    t = torch.linspace(0, 1, steps + 1, device=device)
    step_list = get_crash_schedule(t)
    step_list = step_list[step_list < noise_level]

    alpha, sigma = t_to_alpha_sigma(step_list[-1])
    noised = torch.randn([1, 2, audio.shape[-1]], device='cuda')
    noised = audio * alpha + noised * sigma
    noise = noised

  # Denoise
  if sampler_type == "v-iplms":
    return sampling.iplms_sample(model_fn, noised, step_list.flip(0)[:-1], {})

  if sampler_type == "v-ddim":
    return sampling.sample(model_fn, noise, step_list.flip(0)[:-1], 0, {})


class DDModel:
  def __init__(self, sample_size, sr, custom_ckpt_path):
    self.sample_size = sample_size
    self.sample_rate = sr 
    self.latent_dim = 0  
    self.ckpt_path = custom_ckpt_path  
    self.augs = torch.nn.Sequential(
                Stereo()
                )         

    class Object(object):
      pass

    args = Object()
    args.sample_size = self.sample_size
    args.sample_rate = self.sample_rate
    args.latent_dim = self.latent_dim
    
    print("Creating the model...")
    self.model = DiffusionUncond(args)
    self.model.load_state_dict(torch.load(self.ckpt_path)["state_dict"])
    self.model = self.model.requires_grad_(False).to("cpu")
    print("Model created")

    # # Remove non-EMA
    del self.model.diffusion
    self.model_fn = self.model.diffusion_ema
    self.sampler_type = "v-ddim" 
    self.eta = 0

  def sample(audio_sample, steps, init_strength):
    noise_level = 1.0-init_strength
    stereo_audio = self.augs(audio_sample).unsqueeze(0)
    generated = resample(self.model_fn, stereo_audio, steps, sampler_type=self.sampler_type, noise_level=noise_level)

    return generated
