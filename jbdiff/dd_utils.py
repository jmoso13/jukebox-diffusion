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
import torchaudio


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


def resample(model_fn, audio, noise, steps=100, sampler_type="v-ddim", noise_level = 1.0):
  #Noise the input
  if sampler_type.startswith("v-"):
    t = torch.linspace(0, 1, steps + 1, device='cuda')
    step_list = get_crash_schedule(t)
    step_list = step_list[step_list < noise_level]

    alpha, sigma = t_to_alpha_sigma(step_list[-1])
    noised = noise
    noised = audio * alpha + noised * sigma
    noise_sample = noised

  # Denoise
  if sampler_type == "v-iplms":
    return sampling.iplms_sample(model_fn, noised, step_list.flip(0)[:-1], {})

  if sampler_type == "v-ddim":
    return sampling.sample(model_fn, noise_sample, step_list.flip(0)[:-1], 0, {})


class DDModel:
  def __init__(self, sample_size, sr, custom_ckpt_path, sampler_type="v-ddim"):
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
    # TODO update arg to pass dd sample size
    args.sample_size = 65536
    args.sample_rate = self.sample_rate
    args.latent_dim = self.latent_dim
    
    print("Creating the Dance Diffusion model...")
    self.model = DiffusionUncond(args)
    self.model.load_state_dict(torch.load(self.ckpt_path)["state_dict"])
    self.model = self.model.requires_grad_(False).to("cpu")
    print("Model created")

    # # Remove non-EMA
    del self.model.diffusion
    self.sampler_type = sampler_type
    self.eta = 0

  def sample(self, audio_sample, steps, init_strength, noise):
    # TODO: pass dd sample size as an arg
    og_dd_sample_size = 65536
    noise_level = 1.0-init_strength
    stereo_audio = self.augs(audio_sample.squeeze(0)).unsqueeze(0)
    assert stereo_audio.shape[2] == noise.shape[2]
    pad_length = og_dd_sample_size - stereo_audio.shape[2]
    pad = torch.zeros((1, 2, pad_length)).to('cuda')
    padded_audio = torch.cat([stereo_audio, pad], dim=2)
    padded_noise = torch.cat([noise, pad], dim=2)
    self.model = self.model.to('cuda')
    generated = resample(self.model.diffusion_ema, padded_audio, padded_noise, steps, sampler_type=self.sampler_type, noise_level=noise_level)
    self.model = self.model.to('cpu')
    tmp_saves = {'padded_audio':padded_audio, 'padded_noise':padded_noise, 'generated':generated}
    for fn, save in tmp_saves.items():
      final_audio = rearrange(save, 'b c t -> c (b t)')
      audio_fn = os.path.join('/home/ubuntu/sampling_trials/tmp_save/dd_samples', f"{fn}.wav")
      final_audio = final_audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
      torchaudio.save(audio_fn, final_audio, self.sample_rate)
    generated = generated[:,:,:-pad_length]

    return generated
