import os
import argparse
from jbdiff.utils import read_yaml_file, parse_diff_conf, make_jb, JBDiffusion, load_aud, get_base_noise, Sampler, get_final_audio_container, save_final_audio, combine_wav_files, combine_png_files
import wave
from glob import glob
import numpy as np
import time
import torch as t

#----------------------------------------------------------------------------

# Change config file to change hyperparams
CONFIG_FILE = 'jbdiff-sample-v1.yaml'

# Main function
def run(*args, **kwargs):
  # Load conf file
  conf = read_yaml_file(CONFIG_FILE)

  # Load VQVAE args from conf
  vqvae_conf = conf['model']['vqvae']
  context_mult = vqvae_conf['context_mult']
  batch_size = vqvae_conf['batch_size']
  aug_shift = vqvae_conf['aug_shift']
  base_tokens = vqvae_conf['base_tokens']

  # Load args from command line
  sr = 44100
  token_multiplier = kwargs['token_multiplier']
  seconds_length = kwargs['seconds_length']
  levels = [int(level) for level in kwargs['levels']]
  print('levels: ', levels)
  update_lowest_context = kwargs['update_lowest_context']
  # Check init audio
  init_audio = kwargs['init_audio']
  if init_audio is not None:
    with wave.open(init_audio, 'rb') as wav_file:
      init_num_frames = wav_file.getnframes()
      init_sr = wav_file.getframerate()
      assert init_sr == sr, f"init wav file must be {sr} sample rate to work with JBDiffusion"
      seconds_length = float(init_num_frames)/float(init_sr)
  init_strength = kwargs['init_strength']
  # Check context audio
  context_audio = kwargs['context_audio']
  if context_audio is not None:
    with wave.open(context_audio, 'rb') as wav_file:
      context_num_frames = wav_file.getnframes()
      context_sr = wav_file.getframerate()
      assert context_sr == sr, f"context wav file must be {sr} sample rate to work with JBDiffusion"
  else:
    context_num_frames = sr*seconds_length
    context_sr = sr
  # Noise Params
  noise_seed = kwargs['noise_seed']
  noise_style = kwargs['noise_style'].lower()
  noise_step_size = kwargs['noise_step_size']
  dd_noise_seed = kwargs['dd_noise_seed']
  dd_noise_style = kwargs['dd_noise_style'].lower()
  dd_noise_step_size = kwargs['dd_noise_step_size']
  # Direc params
  save_dir = kwargs['save_dir']
  project_name = kwargs['project_name']

  # Adapt command line args
  use_dd = kwargs['use_dd']
  levels = list(reversed(sorted([l for l in levels if l in (0,1,2)])))
  current_epoch_seconds = int(time.time())
  rotating_seed = current_epoch_seconds%31556952
  rng = np.random.RandomState(rotating_seed)
  if noise_seed is None:
    noise_seed = rng.randint(0, 100000000)
  if dd_noise_seed is None:
    dd_noise_seed = rng.randint(0, 100000000)

  # Set up directories
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  project_name = f"{project_name}_{noise_seed:08d}_{dd_noise_seed:08d}"
  if os.path.exists(os.path.join(save_dir, project_name)):
    num_paths = len(glob(os.path.join(save_dir,f"{project_name}_*")))
    project_name = f"{project_name}_{num_paths:04d}"
  save_dir = os.path.join(save_dir, project_name)
  os.mkdir(save_dir)

  # Load Sampling Args
  sampling_conf = conf['sampling']['diffusion']
  sampling_conf[2]['init_strength'] = init_strength

  # Load diffusion and vqvae models
  diffusion_models = dict()
  for level in levels:
    # Load VQ-VAEs
    vqvae, _, _ = make_jb(".", level, batch_size, base_tokens, context_mult, aug_shift, num_workers=12, train=False)
    # Load Diff Models
    diffusion_conf = conf['model']['diffusion'][level]
    diffusion_conf = parse_diff_conf(diffusion_conf)
    diffusion_conf['embedding_max_length'] = context_mult*base_tokens
    diffusion_models[level] = JBDiffusion(vqvae=vqvae, level=level, diffusion_kwargs=diffusion_conf).to('cpu')
    # Load ckpt state
    diffusion_models[level].load_state_dict(t.load(sampling_conf[level]["ckpt_loc"])["state_dict"])
    diffusion_models[level] = diffusion_models[level].requires_grad_(False).to("cpu")

  # Check that all are in eval
  for level in levels:
    diffusion_models[level].eval()
  for k,v in diffusion_models.items():
    assert not v.diffusion.training
    assert not v.vqvae.training
    print(f"Level {k} VQVAE on device: {next(v.vqvae.parameters()).device}")
    print(f"Level {k} Diffusion Model on device: {next(v.diffusion.parameters()).device}")

  # Setup for Sampling
  level_mults = {0:8, 1:32, 2:128}
  lowest_sample_window_length = token_multiplier*base_tokens*level_mults[levels[0]]
  num_window_shifts = int((seconds_length*sr)//lowest_sample_window_length)
  leftover_window = round(seconds_length*sr) - num_window_shifts*lowest_sample_window_length
  if leftover_window > 0:
    num_window_shifts += 1
    pad = lowest_sample_window_length - leftover_window
  else:
    pad = None

  # Init contexts
  context_windows = dict()
  for level in levels:
    diffusion_models[level] = diffusion_models[level].to('cuda')
    context_windows[level] = diffusion_models[level].get_init_context(context_audio, level_mults, context_num_frames, base_tokens, context_mult, context_sr)
    diffusion_models[level] = diffusion_models[level].to('cpu')

  # Init noise
  noise = get_base_noise(num_window_shifts, token_multiplier*base_tokens, noise_seed, style=noise_style)

  # Load Init Audio and Init Final Audio Container
  if init_audio is not None:
    init_audio = load_aud(init_audio, sr, 0, init_num_frames, pad=pad)
    print(f'init_audio shape: {init_audio.shape}\ndivided by {num_window_shifts} == {init_audio.shape[1]/num_window_shifts}')
  
  print(f'noise shape: {noise.shape}\ndivided by {num_window_shifts} == {noise.shape[2]/num_window_shifts}')

  # Final Audio Container
  final_audio_container = get_final_audio_container(lowest_sample_window_length, num_window_shifts)

  # Define sampling args
  class SamplingArgs:
    def __init__(self):
      self.levels = levels
      self.level_mults = level_mults
      self.base_tokens = base_tokens
      self.token_multiplier = token_multiplier
      self.context_mult = context_mult
      self.save_dir = save_dir
      self.sr = sr
      self.use_dd = use_dd
      self.sampling_conf = sampling_conf
      self.xfade_style = sampling_conf['dd']['xfade_style']
      self.dd_noise_seed = dd_noise_seed
      self.dd_noise_style = dd_noise_style
      self.dd_noise_step = dd_noise_step_size

  # Load Sampler
  sample_args = SamplingArgs()
  sampler = Sampler(cur_sample=0, 
                    diffusion_models=diffusion_models, 
                    context_windows=context_windows, 
                    final_audio_container=final_audio_container, 
                    sampling_args=sample_args
                  )

  for shift in range(num_window_shifts):
    sampler.sample_level( step=shift, 
                          steps=num_window_shifts, 
                          level_idx=0, 
                          base_noise=noise, 
                          base_init=init_audio
                        )
    if update_lowest_context:
      sampler.update_context_window(levels[0])
      from einops import rearrange
      context_q = rearrange(sampler.context_windows[levels[0]].clone().to('cpu'), 'b t c -> b c t')
      decoded_context = diffusion_models[levels[0]].decode(context_q)
      decoded_context = rearrange(decoded_context, 'b t c -> b c t')
      save_final_audio(decoded_context, '/home/ubuntu/sampling_trials/tmp_save/context', sr)

  if pad is not None:
    final_audio = sampler.final_audio_container[:,:,:-pad].clone()
  else:
    final_audio = sampler.final_audio_container.clone()
  save_final_audio(final_audio, save_dir, sr)

  for level in levels:
    combine_wav_files(save_dir, level)
    fps = round(sr/(token_multiplier*base_tokens)/level_mults[level], 3)
    combine_png_files(save_dir, level, fps)

  print(f"Files can be found in {save_dir}")


#----------------------------------------------------------------------------


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _path_exists(p):
  if p is not None:
    if os.path.exists(p):
      return p
    else:
      raise argparse.ArgumentTypeError('Input path does not exist.')
  return p


#----------------------------------------------------------------------------


_examples = '''examples:

  # Sample for 30s using all levels with no init audio conditioned on song_a.wav, save results to a directory called results/
  python sample.py --seconds-length 30 --context-audio song_a.wav --save-dir results --project-name jbdiff_fun --levels 012

  # Sample for length of init audio song_b.wav using song_a.wav as context, use only levels 2 & 1 and use token-multiplier of 4, both of these will speed up generation, also change the dd-noise-style to 'walk'
  python sample.py --init-audio song_b.wav --init-strength 0.15 --context-audio song_a.wav --save-dir results --project-name jbdiff_fun --levels 12 --dd-noise-style walk --token-multiplier 4

'''


def main():
  parser = argparse.ArgumentParser(
    description = 'Sample from JBDiffusion', 
    epilog=_examples, 
    formatter_class=argparse.RawDescriptionHelpFormatter
    )
  # parser.add_argument('--log-to-wandb', help='T/F whether to log to weights and biases', default=False, metavar='BOOL', type=_str_to_bool)
  parser.add_argument('--seconds-length', help='Length in seconds of sampled audio', default=12, type=int)
  parser.add_argument('--init-audio', help='Optionally provide location of init audio to alter using diffusion', default=None, metavar='FILE', type=_path_exists)
  parser.add_argument('--init-strength', help='The init strength alters the range of time conditioned steps used to diffuse init audio, float between 0-1, 1==return original image, 0==diffuse from noise', default=0.0, type=float)
  parser.add_argument('--context-audio', help='Provide the location of context audio', required=True, metavar='FILE', type=_path_exists)
  parser.add_argument('--save-dir', help='Name of directory for saved files', required=True, type=str)
  parser.add_argument('--levels', help='Levels to use for upsampling', default=[0,1,2,'dd'], type=list)
  parser.add_argument('--project-name', help='Name of project', default='JBDiffusion', type=str)
  parser.add_argument('--noise-seed', help='Random seed to use for sampling base layer of Jukebox Diffusion', default=None, type=int)
  parser.add_argument('--noise-style', help='How the random noise for generating base layer of Jukebox Diffusion progresses: random, constant, region, walk', default='random', type=str)
  parser.add_argument('--dd-noise-seed', help='Random seed to use for sampling Dance Diffusion', default=None, type=int)
  parser.add_argument('--dd-noise-style', help='How the random noise for generating in Dance Diffusion progresses: random, constant, region, walk', default='random', type=str)
  parser.add_argument('--noise-step-size', help='How far to wander around init noise, should be between 0-1, if set to 0 will act like constant noise, if set to 1 will act like random noise', default=0.6, type=float)
  parser.add_argument('--dd-noise-step-size', help='How far to wander around init DD noise, should be between 0-1, if set to 0 will act like constant noise, if set to 1 will act like random noise', default=0.4, type=float)
  parser.add_argument('--token-multiplier', help='Multiplier for base_tokens', default=1, type=int)
  parser.add_argument('--use-dd', help='whether or not to use dd', default=True, type=bool)
  parser.add_argument('--update-lowest-context', help='whether or not to update lowest context', default=False, type=bool)
  args = parser.parse_args()


  run(**vars(args))


#----------------------------------------------------------------------------


if __name__ == "__main__":
    main()


#----------------------------------------------------------------------------
