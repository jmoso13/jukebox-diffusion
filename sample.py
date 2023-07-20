import os
import argparse
from jbdiff.utils import read_yaml_file, parse_diff_conf, make_jb, JBDiffusion, load_aud_file
import wave

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
  seconds_length = kwargs['seconds_length']
  init_audio = kwargs['init_audio']
  if init_audio is not None:
    with wave.open(init_audio, 'rb') as wav_file:
      init_num_frames = wav_file.getnframes()
      init_sr = wav_file.getframerate()
      assert init_sr == 44100, "init wav file must be 44100 sample rate to work with JBDiffusion"
      seconds_length = float(init_num_frames)/float(init_sr)
  init_strength = kwargs['init_strength']
  context_audio = kwargs['context_audio']
  if context_audio is not None:
    with wave.open(context_audio, 'rb') as wav_file:
      context_num_frames = wav_file.getnframes()
      context_sr = wav_file.getframerate()
      assert context_sr == 44100, "context wav file must be 44100 sample rate to work with JBDiffusion"
  save_dir = kwargs['save_dir']
  levels = kwargs['levels']

  # Adapt command line args
  use_dd = 'dd' in levels
  levels = list(reversed(sorted([l for l in levels if l in (0,1,2)])))

  # Load Sampling Args
  sampling_conf = conf['sampling']['diffusion']

  # Load diffusion and vqvae models
  hps = dict()
  diffusion_models = dict()
  for level in levels:
    # Load VQ-VAEs
    vqvae, _, hps[level] = make_jb(audio_dir, level, batch_size, base_tokens, context_mult, aug_shift, num_workers, train=False)
    print(f'Sample length for level {level}: {hps[level].sample_length}')
    # Load Diff Models
    diffusion_conf = conf['model']['diffusion'][level]
    diffusion_conf = parse_diff_conf(diffusion_conf)
    diffusion_conf['embedding_max_length'] = context_mult*base_tokens
    diffusion_models[level] = JBDiffusion(vqvae=vqvae, level=level, diffusion_kwargs=diffusion_conf).to('cpu')
    # Load ckpt state

  # Check that all are in eval
  for level in levels:
    diffusion_models.eval()
  for k,v in diffusion_models:
    assert not v.diffusion.training
    assert not v.vqvae.training
    print(f"Level {k} VQVAE on device: {v.vqvae.device}")
    print(f"Level {k} Diffusion Model on device: {v.diffusion.device}")

  # Sample
  level_mults = {0:8, 1:32, 2:128}
  lowest_sample_window_length = hps.sample_length
  num_window_shifts = int((seconds_length*hps.sr)//lowest_sample_window_length)
  leftover_window = round(seconds_length*hps.sr) - num_window_shifts*lowest_sample_window_length

  # Init contexts
  context_windows = dict()
  for level in levels:
    diffusion_models[level] = diffusion_models[level].to('cuda')
    context_windows[level] = diffusion_models[level].get_init_context(context_audio, level_mults, context_num_frames, base_tokens, context_mult, context_sr)
    diffusion_models[level] = diffusion_models[level].to('cpu')

  for shift in num_window_shifts:
    sample_level(diffusion_models, levels, 0, level_mults)
    
def sample_level(diffusion_models, levels, level_idx, level_mults, context_windows):
  level = levels[level_idx]
  diffusion_models[level] = diffusion_models[level].to('cuda')
  # sample
  diffusion_models[level] = diffusion_models[level].to('cpu')
  save_sample

  if level == levels[-1]:
    sample_dd()
    crossfade()

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
  # parser.add_argument('--lowest-level-pkl', help='Location of lowest level network pkl for use in sampling', default=None, metavar='FILE', type=_path_exists)
  # parser.add_argument('--middle-level-pkl', help='Location of middle level network pkl for use in sampling', default=None, metavar='FILE', type=_path_exists)
  # parser.add_argument('--highest-level-pkl', help='Location of highest level network pkl for use in sampling', default=None, metavar='FILE', type=_path_exists)
  args = parser.parse_args()


  run(**vars(args))


#----------------------------------------------------------------------------


if __name__ == "__main__":
    main()


#----------------------------------------------------------------------------
