import jukebox
import torch as t
import librosa
import os
import numpy as np
import math
import jukebox.utils.dist_adapter as dist
from torch.utils.data import Dataset
from torch import Tensor
from jukebox.utils.dist_utils import print_all
from jukebox.utils.io import get_duration_sec, load_audio
from jukebox.data.labels import Labeller
from jukebox.make_models import make_vqvae, MODELS
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.utils.dist_utils import setup_dist_from_mpi
from torch.utils.data import DataLoader
from jukebox.utils.audio_utils import audio_preprocess
rank, local_rank, device = setup_dist_from_mpi()
from einops import rearrange, repeat
# t.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import matplotlib.pyplot as plt
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import sys
import torchaudio
import torchaudio.transforms as transforms
import wandb
import tqdm
import yaml
import importlib
import subprocess
from jbdiff.dd_utils import DDModel
import auraloss


class FilesAudioDataset(Dataset):
    '''
    Lifted from OpenAI Jukebox Repo, altered to return context as well as training batch

    Params
    ______

    - hps: hyperparameters built using setup_hyparams from jukebox repo

    '''
    def __init__(self, hps, context_mult):
        super().__init__()
        self.sr = hps.sr
        self.channels = hps.channels
        self.min_duration = hps.min_duration or math.ceil(hps.sample_length / hps.sr)
        self.max_duration = hps.max_duration or math.inf
        self.sample_length = hps.sample_length
        assert hps.sample_length / hps.sr < self.min_duration, f'Sample length {hps.sample_length} per sr {hps.sr} ({hps.sample_length / hps.sr:.2f}) should be shorter than min duration {self.min_duration}'
        self.aug_shift = hps.aug_shift
        self.labels = hps.labels
        self.init_dataset(hps)
        self.context_mult = context_mult

    def filter(self, files, durations):
        # Remove files too short or too long
        keep = []
        for i in range(len(files)):
            if durations[i] / self.sr < self.min_duration:
                continue
            if durations[i] / self.sr >= self.max_duration:
                continue
            keep.append(i)
        print_all(f'self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}')
        print_all(f"Keeping {len(keep)} of {len(files)} files")
        self.files = [files[i] for i in keep]
        self.durations = [int(durations[i]) for i in keep]
        self.cumsum = np.cumsum(self.durations)

    def init_dataset(self, hps):
        # Load list of files and starts/durations
        files = librosa.util.find_files(f'{hps.audio_files_dir}', ['mp3', 'opus', 'm4a', 'aac', 'wav'])
        print_all(f"Found {len(files)} files. Getting durations")
        cache = dist.get_rank() % 8 == 0 if dist.is_available() else True
        durations = np.array([get_duration_sec(file, cache=cache) * self.sr for file in files])  # Could be approximate
        self.filter(files, durations)

        if self.labels:
            self.labeller = Labeller(hps.max_bow_genre_size, hps.n_tokens, self.sample_length, v3=hps.labels_v3)

    def get_index_offset(self, item):
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length//2
        shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0
        offset = item * self.sample_length + shift # Note we centred shifts, so adding now
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], f'Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}'
        index = np.searchsorted(self.cumsum, midpoint)  # index <-> midpoint of interval lies in this song
        start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[index] # start and end of current song
        assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"
        if offset > end - self.sample_length: # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        elif offset < start: # Going under song
            offset = min(end - self.sample_length, offset + half_interval)  # Now should fit
        assert start <= offset <= end - self.sample_length, f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"
        offset = offset - start
        context_offset = max(0, offset - self.sample_length*self.context_mult)
        return index, offset, context_offset

    def get_metadata(self, filename, test):
        """
        Insert metadata loading code for your dataset here.
        If artist/genre labels are different from provided artist/genre lists,
        update labeller accordingly.

        Returns:
            (artist, genre, full_lyrics) of type (str, str, str). For
            example, ("unknown", "classical", "") could be a metadata for a
            piano piece.
        """
        return None, None, None

    def get_song_chunk(self, index, offset, test=False):
        filename, total_length = self.files[index], self.durations[index]
        data, sr = load_audio(filename, sr=self.sr, offset=offset, duration=self.sample_length)
        assert data.shape == (self.channels, self.sample_length), f'Expected {(self.channels, self.sample_length)}, got {data.shape}'
        if self.labels:
            artist, genre, lyrics = self.get_metadata(filename, test)
            labels = self.labeller.get_label(artist, genre, lyrics, total_length, offset)
            return data.T, labels['y']
        else:
            return data.T
    
    def get_song_context_chunk(self, index, offset, context_offset, test=False):
        filename, total_length = self.files[index], self.durations[index]
        data, sr = load_audio(filename, sr=self.sr, offset=offset, duration=self.sample_length)
        context = repeat(np.zeros(data.shape), 'c t -> c (repeat t)', repeat = self.context_mult)
        context_data, _ = load_audio(filename, sr=self.sr, offset=context_offset, duration = self.sample_length*self.context_mult)
        length = int(offset - context_offset)
        context_data = context_data[:, :length]
        if length > 0:
            context[:, -length:] += context_data
        assert data.shape == (self.channels, self.sample_length), f'Expected {(self.channels, self.sample_length)}, got {data.shape}'
        if self.labels:
            artist, genre, lyrics = self.get_metadata(filename, test)
            labels = self.labeller.get_label(artist, genre, lyrics, total_length, offset)
            return data.T, labels['y']
        else:
            return data.T, context.T

    def get_item(self, item, test=False):
        index, offset, context_offset = self.get_index_offset(item)
        return self.get_song_context_chunk(index, offset, context_offset, test)

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, item):
        return self.get_item(item)


def make_jb(train_data, level, batch_size, base_tokens, context_mult, aug_shift, num_workers, train=True):
    '''
    Constructs vqvae model as well as the dataloader for batching

    Params
    _______
    train_data: (str) location of audio files to train on
    level: (int) level of vqvae latent codes being trained on
    batch_size: (int) number of examples per batch for training
    base_tokens: (int) length of token sequence for diffusion on each level, multiply by 'level_mult' to translate from token length to sample length 
    aug_shift: (bool) if True, adds random cropping to each training example, if false, sequentially cuts up audio data
    num_workers: (int) number of workers for the dataloader, depends on training machine

    Returns
    _______
    Tuple of
    vqvae: (Jukebox) instance of the vqvae encoder/decoder for retrieving latent codes
    dataloader: (DataLoader) instance of DataLoader for loading training examples from directory of audio files
    hps: (dict) dictionary of hyperparameters
    '''
    base_model = "5b"    
    level_mult = 8 if level == 0 else 32 if level == 1 else 128
    sample_length = base_tokens*level_mult
    vqvae, *priors = MODELS[base_model]
    hps = setup_hparams(vqvae, dict(sample_length=sample_length, audio_files_dir=train_data, labels=False, train_test_split=0.8, aug_shift=aug_shift, bs=batch_size))
    if train:
        dataset = FilesAudioDataset(hps, context_mult)
        dataloader = DataLoader(dataset, batch_size=hps.bs, num_workers=num_workers, pin_memory=False, drop_last=True)
    else:
        dataloader = None
    vqvae = make_vqvae(hps, device)
    return vqvae, dataloader, hps


def norm_pre(x, vqvae, level):
    '''
    Normalizes latent codes by subtracting by mean and multiply by std of level's quantize codebook
    '''
    og_k = vqvae.bottleneck.level_blocks[level].k.detach()
    mean = og_k.mean()
    std = og_k.std()
    return (x-mean)/std


def norm_post(x, vqvae, level):
    '''
    Returns latent codes from normalized version to raw format
    '''
    og_k = vqvae.bottleneck.level_blocks[level].k.detach()
    mean = og_k.mean()
    std = og_k.std()
    return x*std+mean


def batch_preprocess(x, vqvae, level):
    '''
    Takes audio and preprocesses it for use in diffusion
    '''
    class tmp_hps:
        def __init__(self):
            self.aug_blend=False
            self.channels=2
    hps = tmp_hps()
    x = audio_preprocess(x, hps)
    x = vqvae.preprocess(x)
    z_q, x_q = vqvae.bottleneck.level_blocks[level](vqvae.encoders[level](x)[-1], update_k=False)[:2]
    x_q = norm_pre(x_q, vqvae, level)
    return z_q, x_q


def batch_postprocess(x, vqvae, level, quantize=True):
    '''
    Takes diffused codes and postprocesses it back to audio
    '''
    x = norm_post(x, vqvae, level)
    z = None
    x_q = None
    if quantize:
        z, x_q = vqvae.bottleneck.level_blocks[level](x, update_k=False)[:2]
        x = vqvae.decoders[level]([x_q], all_levels=False)
    else:
        x = vqvae.decoders[level]([x], all_levels=False)
    return x, z, x_q


@t.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class CombinedLoss(t.nn.Module):
    def __init__(self, sample_rate):
        super(CombinedLoss, self).__init__()
        self.sample_rate = sample_rate
        self.mse_loss = t.nn.functional.mse_loss
        self.sdstft = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=self.sample_rate, 
                                                            fft_sizes = [256, 128, 64, 32, 16],
                                                            hop_sizes = [64, 32, 16, 8, 4],
                                                            win_lengths = [256, 128, 64, 32, 16],
                                                            w_sc = 1.0,
                                                            w_log_mag = 1.0,
                                                            w_lin_mag = 0.0,
                                                            w_phs = 0.33
                                                            )

    def forward(self, inputs, targets):
        mse_loss = self.mse_loss(inputs, targets)
        sdstft_loss = self.sdstft(inputs, targets)
        loss = 0.1*mse_loss + 0.9*sdstft_loss
        return (loss, mse_loss, sdstft_loss)


class JBDiffusion(pl.LightningModule):
    '''
    JBDiffusion class to be trained

    Init Params
    ___________
    - vqvae: (Jukebox) instance of the vqvae encoder/decoder for retrieving latent codes
    - level: (int) level of vqvae latent codes to train on (2->0)
    - diffusion_kwargs: (dict) dict of diffusion kwargs
    '''
    def __init__(self, vqvae, level, diffusion_kwargs):
        super().__init__()

        self.level = level
        self.loss_fn = CombinedLoss(sample_rate=44100)
        self.diffusion = DiffusionModel(loss_fn=self.loss_fn, **diffusion_kwargs)
        # self.diffusion_ema = deepcopy(self.diffusion)
        # self.ema_decay = global_args.ema_decay
        self.vqvae = vqvae
    
    def configure_optimizers(self):
        return t.optim.Adam([*self.diffusion.parameters()], lr=4e-5)

    def training_step(self, batch, batch_idx):
        # Assure training
        self.diffusion.train()
        assert self.diffusion.training
        # Grab train batch
        x, cond = batch
        # Preprocess batch and conditional audio for diffusion (includes running batch through Jukebox encoder)
        z_q, x_q = batch_preprocess(x, self.vqvae, self.level)
        cond_z, cond_q = batch_preprocess(cond, self.vqvae, self.level)
        cond_q = rearrange(cond_q, "b c t -> b t c")
        with t.cuda.amp.autocast():
            # Step
            loss, mse_loss, sdstft_loss = self.diffusion(x_q, embedding=cond_q, embedding_mask_proba=0.1)

        log_dict = {
            'train/loss': loss.detach(),
            'train/mse_loss': mse_loss.detach(),
            'train/sdstft_loss': sdstft_loss.detach()
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def sample(self, noise, num_steps, init, init_strength, context, context_strength):
        if init is not None:
            start_step = int(init_strength*num_steps)
            sigmas = self.diffusion.sampler.schedule(num_steps + 1, device='cuda')
            sigmas = sigmas[start_step:]
            sigmas = repeat(sigmas, "i -> i b", b=1)
            sigmas_batch = extend_dim(sigmas, dim=noise.ndim + 1)
            alphas, betas = self.diffusion.sampler.get_alpha_beta(sigmas_batch)
            alpha, beta = alphas[0], betas[0]
            x_noisy = alpha*init + beta*noise
            progress_bar = tqdm.tqdm(range(num_steps-start_step), disable=False)

            for i in progress_bar:
                v_pred = self.diffusion.sampler.net(x_noisy, sigmas[i], embedding=context, embedding_scale=context_strength)
                x_pred = alphas[i] * x_noisy - betas[i] * v_pred
                noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
                x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
                progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")

            x_noisy_audio, _, _= batch_postprocess(x_noisy, self.vqvae, self.level)
            x_noisy_audio = rearrange(x_noisy_audio, "b c t -> b t c")

            return x_noisy, x_noisy_audio
        else:
            sample = self.diffusion.sample(
                    noise,
                    embedding=context,
                    embedding_scale=context_strength, 
                    num_steps=num_steps
                    )

            sample_audio, _, _= batch_postprocess(sample, self.vqvae, self.level)
            sample_audio = rearrange(sample_audio, "b c t -> b t c")

            return sample, sample_audio

    def get_init_context(self, context_audio_file, level_mults, context_num_frames, base_tokens, context_mult, sr):
        level_mult = level_mults[self.level]
        context_frames = context_mult*base_tokens*level_mult
        cutoff = context_frames if context_frames <= context_num_frames else context_num_frames
        offset = max(0, int(context_num_frames-context_frames))
        if context_audio_file is not None:
            data, _ = load_audio(context_audio_file, sr=sr, offset=offset, duration=cutoff)
        else:
            data = np.zeros((2, context_frames))
        context = np.zeros((data.shape[0], context_frames))
        context[:, -cutoff:] += data
        context = context.T
        context = t.tensor(np.expand_dims(context, axis=0)).to('cuda', non_blocking=True).detach()
        context_z, context_q = batch_preprocess(context, self.vqvae, self.level)
        context_q = rearrange(context_q, "b c t -> b t c")

        return context_q

    def encode(self, audio):
        return batch_preprocess(audio, self.vqvae, self.level)

    def decode(self, audio_q):
        decoded_audio, _, _ = batch_postprocess(audio_q, self.vqvae, self.level)
        return rearrange(decoded_audio, "b c t -> b t c")

    # def on_before_zero_grad(self, *args, **kwargs):
    #     decay = 0.95 if self.current_epoch < 25 else self.ema_decay
    #     ema_update(self.diffusion, self.diffusion_ema, decay)


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    '''
    Class for demoing during training

    Init Params
    ____________
    - global_args: (DemoArgs class) kwargs for demoing
    '''
    def __init__(self, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.num_demos = global_args.num_demos
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.sample_rate = global_args.sample_rate
        self.last_demo_step = -1
        self.base_samples = global_args.base_samples
        self.base_tokens = global_args.base_tokens
        self.dirpath = global_args.dirpath
        self.embedding_scale = global_args.embedding_scale
        self.context_mult = global_args.context_mult

    @rank_zero_only
    @t.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):        

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        
        # Assert eval mode
        module.diffusion.eval()
        assert not module.diffusion.training
        self.last_demo_step = trainer.global_step

        # Grab batch and process
        x, cond = batch
        z_q, x_q = batch_preprocess(x, module.vqvae, module.level)
        z_cond, cond_q = batch_preprocess(cond, module.vqvae, module.level)
        x_q = x_q[:self.num_demos]
        cond_q = cond_q[:self.num_demos]
        embedding = rearrange(cond_q, "b c t -> b t c")
        try:
            # Define noise
            noise = t.randn([self.num_demos, 64, self.base_tokens]).to(device)
            # Number of times to diffuse to get to demo length
            hops = self.demo_samples//self.base_samples
            # Full audio container
            full_fakes = t.tensor(repeat(np.zeros((self.num_demos, 1, self.base_samples)), 'b c t -> b c (repeat t)', repeat = hops+5), device=device)
            # Include conditioned and noisy audio in the demo
            x_a, _, n_x = batch_postprocess(x_q, module.vqvae, module.level)
            cond_a, _, n_c =  batch_postprocess(cond_q, module.vqvae, module.level)
            full_fakes[:, :, :self.base_samples*5] += t.cat([cond_a, x_a, cond_a], dim = 2)
            # Diffuse
            for hop in tqdm.tqdm(range(hops)):
                fakes = module.diffusion.sample(
                        noise.float(),
                        embedding=embedding.float(),
                        embedding_scale=self.embedding_scale, 
                        num_steps=self.demo_steps
                      )
                # Add diffused example to demo
                fakes, sample_z, sample_q = batch_postprocess(fakes.detach(), module.vqvae, module.level)
                sampled = rearrange(norm_pre(sample_q, module.vqvae, module.level), 'b c t -> b t c')
                # Update embedding
                embedding = t.cat([*embedding.chunk(self.context_mult, dim=1)[1:], sampled], dim=1)
                full_fakes[:, :, self.base_samples*(hop+5):self.base_samples*(hop+6)] += fakes
                # Sample randomly new noise
                noise = t.randn([self.num_demos, 64, self.base_tokens]).to(device)

            # Put the demos together
            full_fakes = rearrange(full_fakes, 'b d n -> d (b n)')

            log_dict = {}

            # Create path for demos
            demo_path = os.path.join(self.dirpath, 'demo_wavs')
            if not os.path.exists(demo_path):
                os.mkdir(demo_path)
            # Save & log demo
            filename = os.path.join(demo_path, f'demo_{trainer.global_step:08}.wav')
            full_fakes = full_fakes.clamp(-1, 1).mul(32767).to(t.int16).cpu()
            torchaudio.save(filename, full_fakes, self.sample_rate)

            log_dict[f'demo'] = wandb.Audio(filename,
                                              sample_rate=self.sample_rate,
                                              caption=f'Demo')

            # log_dict[f'demo_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))

            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)
            raise e


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
    return data


def parse_diff_conf(diff_conf):
    new_conf = {k:(get_obj_from_str(v) if '_t' in k else v) for k,v in diff_conf.items()}
    return new_conf


def load_aud(fn, sr, offset, duration, pad=None):
    audio, _ = load_audio(fn, sr=sr, offset=offset, duration=duration)
    if pad is not None:
        padded = np.zeros((audio.shape[0], pad))
        audio = np.concatenate([audio, padded], axis=1)
    audio = audio.T
    return t.tensor(np.expand_dims(audio, axis=0)).to('cuda', non_blocking=True).detach()


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))


def custom_random_generator(seed):
    """
    Create a custom random number generator with the given seed.

    Args:
        seed (int): Random seed.

    Returns:
        torch.Generator: Custom random number generator.
    """
    generator = t.Generator()
    generator.manual_seed(seed)
    return generator


def get_base_noise(num_window_shifts, base_tokens, noise_seed, style='random', noise_step_size=0.05):
    rng = custom_random_generator(noise_seed)
    home_noise_weight = math.sqrt(1 - noise_step_size**2)
    if style == 'random':
        return t.randn([1, 64, num_window_shifts*base_tokens], generator=rng).to(device)
    elif style == 'constant':
        r_noise = t.randn([1, 64, base_tokens], generator=rng)
        return t.cat([r_noise for _ in range(num_window_shifts)], dim=2).to(device)
    elif style == 'region':
        home_noise = t.randn([1, 64, base_tokens], generator=rng)
        return t.cat([home_noise]+[home_noise_weight*home_noise+noise_step_size*t.randn([1, 64, base_tokens], generator=rng) for _ in range(num_window_shifts-1)], dim=2).to(device)
    elif style == 'walk':
        home_noise = t.randn([1, 64, base_tokens], generator=rng)
        cumulative_tensors = [home_noise]
        new_noise = home_noise
        for i in range(num_window_shifts-1):
            new_noise = home_noise_weight*new_noise + noise_step_size*t.randn([1, 64, base_tokens], generator=rng)
            cumulative_tensors += [new_noise]
        return t.cat(cumulative_tensors, dim=2).to(device)
    else:
        raise Exception("Noise style must be either 'constant', 'random', 'region', or 'walk'")


def get_final_audio_container(lowest_sample_window_length, num_window_shifts):
    return t.zeros((1, 2, lowest_sample_window_length*num_window_shifts))


def save_final_audio(final_audio, save_dir, sr):
    final_audio = rearrange(final_audio, 'b c t -> c (b t)')
    audio_fn = os.path.join(save_dir, f"final_audio.wav")
    final_audio = final_audio.clamp(-1, 1).mul(32767).to(t.int16).cpu()
    torchaudio.save(audio_fn, final_audio, sr)


def combine_wav_files(save_dir, level):
    # Get input_direc
    input_directory = os.path.join(save_dir, str(level))

    # Get a list of all .wav files in the input directory
    wav_files = sorted([file for file in os.listdir(input_directory) if file.endswith(".wav")])

    # Create a list of input file paths
    input_files = [os.path.join(input_directory, file) for file in wav_files]

    # Define output file loc
    output_file = os.path.join(save_dir, f"{level}.wav")

    # Create a text file containing the list of input WAV files
    with open(os.path.join(input_directory, "input.txt"), "w") as f:
        for file in input_files:
            f.write(f"file '{file}'\n")

    # Call FFmpeg to concatenate the WAV files using the concat demuxer
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "concat", "-safe", "0", "-i", os.path.join(input_directory, "input.txt"),
        "-c", "copy", output_file
    ]
    subprocess.run(ffmpeg_cmd)


def combine_png_files(save_dir, level, fps):
    # Get input_direc
    input_directory = os.path.join(save_dir, str(level))

    # Concat regex for pngs
    all_pngs = os.path.join(input_directory, '*.png')

    # Define output file loc
    output_file = os.path.join(save_dir, f"{level}.mp4")

    # Use ffmpeg to concatenate the audio files
    ffmpeg_command = ["ffmpeg", "-framerate", str(fps), "-pattern_type", "glob", "-i", f"{all_pngs}", "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-preset", "veryslow", output_file]
    subprocess.run(ffmpeg_command)


def combine_video_with_audio(save_dir, level):
    """
    Combines an .mp4 video and .wav audio file into a single video with synchronized audio.
    """
    video_path = os.path.join(save_dir, f"{level}.mp4")
    audio_path = os.path.join(save_dir, f"{level}.wav")
    output_path = os.path.join(save_dir, f"{level}_combined.mp4")

    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]

    subprocess.run(ffmpeg_command)


def wget(url, outputdir='.'):
    res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)


class Sampler:
    def __init__(self, cur_sample, diffusion_models, context_windows, final_audio_container, sampling_args):
        self.cur_sample = cur_sample
        self.sr = sampling_args.sr
        self.use_dd = sampling_args.use_dd
        self.diffusion_models = diffusion_models
        self.levels = sampling_args.levels
        self.level_mults = sampling_args.level_mults
        self.base_tokens = sampling_args.base_tokens
        self.token_multiplier = sampling_args.token_multiplier
        self.context_windows = context_windows
        self.context_mult = sampling_args.context_mult 
        self.final_audio_container = final_audio_container
        self.save_dir = sampling_args.save_dir
        self.sampling_conf = sampling_args.sampling_conf
        self.last_layer_0 = None
        self.xfade_style = sampling_args.xfade_style.lower()
        assert self.xfade_style in ('linear, constant-power'), "chosen xfade_style has to be either 'linear' or 'constant-power', please alter in yaml"
        if self.use_dd:
            self.dd_base_samples = self.base_tokens*self.token_multiplier*self.level_mults[self.levels[-1]]
            self.dd_xfade_samples = self.sampling_conf["dd"]["xfade_samples"]
            self.dd_sample_size = 65536
            self.dd_ckpt = self.sampling_conf["dd"]["ckpt_loc"]
            self.dd_steps = self.sampling_conf["dd"]["num_steps"]
            self.dd_init_strength = self.sampling_conf["dd"]["init_strength"]
            self.dd_noise_rng = custom_random_generator(sampling_args.dd_noise_seed)
            self.dd_noise_style = sampling_args.dd_noise_style
            self.dd_noise_step = sampling_args.dd_noise_step
            self.dd_home_noise_scale = math.sqrt(1 - self.dd_noise_step**2)
            self.dd_effective_length = 0
            while self.dd_effective_length < self.dd_base_samples+self.dd_xfade_samples:
                self.dd_effective_length += self.dd_sample_size
            self.dd_noise = t.randn([1, 2, self.dd_effective_length], generator=self.dd_noise_rng).to(device)
            self.dd_model = DDModel(sample_size=self.dd_sample_size, effective_length=self.dd_effective_length, sr=self.sr, custom_ckpt_path=self.dd_ckpt)
            self.dd_home_noise = self.dd_noise.clone()

    def sample_level(self, step, steps, level_idx, base_noise, base_init):
        level = self.levels[level_idx]
        print(f"sampling level {level} out of levels {self.levels}\nsampling step {step+1} out of {steps} steps on this level")
        # To GPU
        self.diffusion_models[level] = self.diffusion_models[level].to('cuda')
        # Cut up and encode noise & init
        ### THIS IS WHAT NEEDS TO CHANGE FOR SAMPLING ###
        ### GRAB INTERMEDIATE NOISE BASED OFF GENERATOR LIKE BASE NOISE ###
        ### UPSAMPLER PROBLEM ###
        ### UPSAMPLERS STILL IMPLEMENTED HERE ###
        if level < 2:
            cur_noise = base_noise.chunk(steps, dim=1)[step]
            _, noise_enc = self.diffusion_models[level].encode(cur_noise)
        else:
            noise_enc = base_noise.chunk(steps, dim=2)[step]
        if base_init is not None:
            cur_init = base_init.chunk(steps, dim=1)[step]
            _, init_enc = self.diffusion_models[level].encode(cur_init)
        else:
            init_enc = None
        # Grab hps from sampling conf
        num_steps = self.sampling_conf[level]['num_steps']
        init_strength = self.sampling_conf[level]['init_strength']
        embedding_strength = self.sampling_conf[level]['embedding_strength']
        # Sample
        sample, sample_audio = self.diffusion_models[level].sample(noise=noise_enc, 
                                                                  num_steps=num_steps, 
                                                                  init=init_enc, 
                                                                  init_strength=init_strength, 
                                                                  context=self.context_windows[level], 
                                                                  context_strength=embedding_strength
                                                                  )
        self.diffusion_models[level] = self.diffusion_models[level].to('cpu')
        self.save_sample_audio(sample_audio, level)

        if level_idx == len(self.levels)-1:
            # Upsample using Dance Diffusion
            if self.use_dd:
                sample_audio = rearrange(sample_audio, "b t c -> b c t")
                if self.cur_sample == 0:
                    padding = t.zeros((1,1,self.dd_xfade_samples)).to('cuda')
                    dd_init = t.cat([padding, sample_audio], dim=2)
                else:
                    padding = self.last_layer_0[:,:,-self.dd_xfade_samples:]
                    dd_init = t.cat([padding, sample_audio], dim=2)
                # Sample
                dd_sample = self.dd_model.sample(dd_init, self.dd_steps, self.dd_init_strength, self.dd_noise)
                # Insert main chunk
                main_sample = dd_sample[:,:,self.dd_xfade_samples:]
                self.final_audio_container[:,:,self.cur_sample:self.cur_sample+self.dd_base_samples] = main_sample
                # Crossfade if we have audio to fade with
                if self.cur_sample >= self.dd_xfade_samples:
                    fade_in = dd_sample[:,:,:self.dd_xfade_samples]
                    fade_out = self.final_audio_container[:,:,self.cur_sample-self.dd_xfade_samples:self.cur_sample]
                    xfade = self.xfade(fade_out, fade_in)
                    self.final_audio_container[:,:,self.cur_sample-self.dd_xfade_samples:self.cur_sample] = xfade
                self.last_layer_0 = sample_audio
                self.cur_sample += self.token_multiplier*self.base_tokens*self.level_mults[level]
                print('cur sample: ', self.cur_sample)
                self.update_dd_noise()
            return None
        else:
            next_steps = self.level_mults[level]//self.level_mults[self.levels[level_idx+1]]
            for next_step in range(next_steps):
                # Sample next level... 
                # TODO: create version to sample using traditional diffusion rather than upsampling trick
                # Using noise_style either provide frozen sampled noise or newly sampled noise every iter for each level
                self.sample_level(next_step, next_steps, level_idx+1, base_noise=sample_audio, base_init=sample_audio)
                self.update_context_window(self.levels[level_idx+1])
            return None

    def save_sample_audio(self, sample_audio, level):
        # Reshape Audio and Save
        audio = rearrange(sample_audio, 'b t c -> c (b t)')
        level_loc = os.path.join(self.save_dir, str(level))
        if not os.path.exists(level_loc):
            os.mkdir(level_loc)
        audio_fn = os.path.join(level_loc, f"{self.cur_sample:09d}.wav")
        final_audio = audio.clamp(-1, 1).mul(32767).to(t.int16).cpu()
        torchaudio.save(audio_fn, final_audio, self.sr)

        # Create a MelSpectrogram
        # hop_length = round((self.base_tokens*self.level_mults[level])/(.025*self.sr))
        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_mels=16,
            hop_length=512
        )
        # Compute the mel spectrogram
        spec_audio = audio.to('cpu')
        mel_spec = mel_spectrogram(spec_audio)
        # Convert the power spectrogram to decibels
        mel_spec_db = transforms.AmplitudeToDB()(mel_spec)
        # Convert the spectrogram tensor to a NumPy array for visualization
        mel_spec_db_np = mel_spec_db.squeeze(0).numpy()
        # Save the mel spectrogram as an image
        mel_fn = os.path.join(level_loc, f"{self.cur_sample:09d}.png")
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec_db_np, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()
        plt.savefig(mel_fn)
        plt.close()

    def xfade(self, fade_out, fade_in):
        assert fade_out.shape[2] == fade_in.shape[2], "Fades are not the same size, investigate"
        fade_in_match = fade_in.clone().to(fade_out.device)
        num_samples = fade_out.shape[2]
        if self.xfade_style == 'linear':
            fade_weights = t.linspace(0.0, 1.0, num_samples, device=fade_out.device)
        elif self.xfade_style == 'constant-power':
            fade_weights = t.sin((math.pi / 2) * t.linspace(0.0, 1.0, num_samples, device=fade_out.device))
        new_fade_out = fade_out*(1 - fade_weights)
        new_fade_in = fade_in_match*fade_weights
        crossfaded_audio = new_fade_in + new_fade_out
        return crossfaded_audio

    def update_context_window(self, level):
        cur_context = self.context_windows[level]
        cur_context_length = cur_context.shape[1]
        cur_context_sample_length = cur_context_length*self.level_mults[level]
        if self.cur_sample < cur_context_sample_length:
            keep = cur_context_length - self.cur_sample//self.level_mults[level]
            keep = cur_context[:,-keep:, :]
            new_audio = self.final_audio_container[:,:,:self.cur_sample]
            new_audio = rearrange(new_audio, "b c t -> b t c")
            _, new_audio_enc = self.diffusion_models[level].encode(new_audio)
            new_audio_enc = rearrange(new_audio_enc, "b c t -> b t c").to(device)
            self.context_windows[level] = t.cat([keep, new_audio_enc], dim=1)
            assert self.context_windows[level].shape[1] == cur_context_length
        else:
            new_audio = self.final_audio_container[:,:,self.cur_sample-cur_context_sample_length:self.cur_sample]
            new_audio = rearrange(new_audio, "b c t -> b t c")
            _, new_audio_enc = self.diffusion_models[level].encode(new_audio)
            new_audio_enc = rearrange(new_audio_enc, "b c t -> b t c").to(device)
            self.context_windows[level] = new_audio_enc.clone()
            assert self.context_windows[level].shape[1] == cur_context_length

    def update_dd_noise(self):
        if self.dd_noise_style == 'random':
            self.dd_noise = t.randn([1, 2, self.dd_effective_length], generator=self.dd_noise_rng).to(device)
        elif self.dd_noise_style == 'constant':
            pass
        elif self.dd_noise_style == 'region':
            self.dd_noise = self.dd_home_noise_scale*self.dd_home_noise + self.dd_noise_step*t.randn([1, 2, self.dd_effective_length], generator=self.dd_noise_rng).to(device)
        elif self.dd_noise_style == 'walk':
            self.dd_noise = self.dd_home_noise_scale*self.dd_noise + self.dd_noise_step*t.randn([1, 2, self.dd_effective_length], generator=self.dd_noise_rng).to(device)
        else:
            raise Exception("DD noise style must be either 'constant', 'random', 'region', or 'walk'")

        # print('Updated dd noise distance from home: ', t.norm(self.dd_home_noise - self.dd_noise))

