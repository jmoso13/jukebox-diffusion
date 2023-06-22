import jukebox
import torch as t
import librosa
import os
import numpy as np
import math
import jukebox.utils.dist_adapter as dist
from torch.utils.data import Dataset
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
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import sys
import torchaudio
import wandb
import tqdm
import yaml
import importlib


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


def make_jb(train_data, level, batch_size, base_tokens, context_mult, aug_shift, num_workers):
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
    dataset = FilesAudioDataset(hps, context_mult)
    dataloader = DataLoader(dataset, batch_size=hps.bs, num_workers=num_workers, pin_memory=False, drop_last=True)
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
        self.upsampler = self.level in (0,1)
        self.diffusion = DiffusionModel(**diffusion_kwargs)
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
        # Upsampler uses noisy data from level below
        if self.upsampler:
            # Run example through level below back to noisy audio
            _, x_noise_q = batch_preprocess(x, self.vqvae, self.level+1)
            x_noise_audio, _, _ = batch_postprocess(x_noise_q, self.vqvae, self.level+1)
            x_noise_audio = rearrange(x_noise_audio, "b c t -> b t c")
            # Preprocess and encode noisy audio at current level
            _, x_noise = batch_preprocess(x_noise_audio, self.vqvae, self.level)
            xn_q = rearrange(x_noise, "b c t -> b t c")
            with t.cuda.amp.autocast():
                # Step
                loss = self.diffusion(x_q, embedding=xn_q, embedding_mask_proba=0.1)
        else:
            with t.cuda.amp.autocast():
                # Step
                loss = self.diffusion(x_q, embedding=cond_q, embedding_mask_proba=0.1)

        log_dict = {
            'train/loss': loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

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
        # embedding = rearrange(cond_q, "b c t -> b t c")
        try:
            if module.upsampler:
                # If upsampler run audio through the lower level to extract noisy audio
                _, x_noise_q = batch_preprocess(x, module.vqvae, module.level+1)
                x_noise_audio, _, _ = batch_postprocess(x_noise_q, module.vqvae, module.level+1)
                x_noise_audio = rearrange(x_noise_audio, "b c t -> b t c")
                _, noise = batch_preprocess(x_noise_audio, module.vqvae, module.level)
                noise = noise[:self.num_demos]
                embedding = rearrange(noise, "b c t -> b t c")
                # Full audio container
                pad = t.tensor(np.zeros((self.num_demos, 1, self.base_samples)), device=device)
                full_fakes = t.tensor(repeat(np.zeros((self.num_demos, 1, self.base_samples)), 'b c t -> b c (repeat t)', repeat = 6), device=device)
                # Include conditioned and noisy audio in the demo
                x_a, _, n_x = batch_postprocess(x_q, module.vqvae, module.level)
                cond_a, _, n_c =  batch_postprocess(noise, module.vqvae, module.level)
                full_fakes[:, :, :self.base_samples*5] += t.cat([pad, cond_a, x_a, pad, cond_a], dim = 2)
                # Diffuse
                fakes = module.diffusion.sample(
                        noise.float(),
                        embedding=embedding.float(),
                        embedding_scale=self.embedding_scale,
                        num_steps=self.demo_steps 
                      )
                # Add diffused example to demo
                fakes, sample_z, sample_q = batch_postprocess(fakes.detach(), module.vqvae, module.level)
                full_fakes[:, :, self.base_samples*5:] += fakes
            else:
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
                    embedding = t.cat([*embedding.chunk(context_mult, dim=1)[1:], sampled], dim=1)
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

