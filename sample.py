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
  sampling_conf[2]['init_strength'] = init_strength

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

  # Init Sample Windows
  sample_windows

  for shift in num_window_shifts:
    noise = None
    init = None
    sample_level(diffusion_models, levels, 0, level_mults)

class Sampler:
  def __init__(self, cur_sample, diffusion_models, levels, level_mults, context_windows, final_audio_container, save_dir, sampling_conf):
    self.cur_sample = cur_sample
    self.diffusion_models = diffusion_models
    self.levels = levels
    self.level_mults = level_mults
    self.context_windows = context_windows 
    self.final_audio_container = final_audio_container
    self.save_dir = save_dir
    self.sampling_conf = sampling_conf
  
  def sample_level(self, step, steps, level_idx, noise, init):
    level = self.levels[level_idx]
    # To GPU
    self.diffusion_models[level] = self.diffusion_models[level].to('cuda')
    # Cut up and encode noise & init
    cur_noise = noise.chunk(steps, dim=2)[step]
    _, noise_enc = self.diffusion_models[level].encode(cur_noise)
    if init is not None:
      cur_init = init.chunk(steps, dim=2)[step]
      _, init_enc = self.diffusion_models[level].encode(cur_init)
    else:
      init_enc = None
    # Grab hps from sampling conf
    num_steps = self.sampling_conf[level]['num_steps']
    init_strength = self.sampling_conf[level]['init_strength']
    embedding_strength = self.sampling_conf[level]['embedding_strength']
    context = self.context_windows[level]
    # Sample
    sample, sample_audio = self.diffusion_models[level].sample(noise=noise_enc, 
                                                          num_steps=num_steps, 
                                                          init=init_enc, 
                                                          init_strength=init_strength, 
                                                          context=context, 
                                                          context_strength=embedding_strength)
    diffusion_models[level] = self.diffusion_models[level].to('cpu')
    self.save_sample_audio(sample_audio)

    if level_idx == len(levels)-1:
      sample_dd()
      crossfade()

  def save_sample_audio(self, sample_audio):
    pass

  def xfade(audio_1, audio_2)


def sample_check(step, steps, level_idx, cur_sample, noise=np.zeros((1,1,768*2*128)), init=np.zeros((1,1,768*2*128)), levels=[2,1,0], level_mults={0:8, 1:32, 2:128}, context_windows={level:np.zeros((1,768*2,1)) for level in [2,1,0]}, final_audio=np.zeros((1,1,768*2*128))):
  level = levels[level_idx]
  print('cutting up and encoding audio into current level space')
  print(f'cutting noise: {noise.shape} into {steps} steps')
  cur_noise = np.split(noise, steps, axis=2)[step]
  print(f'new noise shape: {cur_noise.shape}')
  print(f'sampling level {level} on step {step}')
  sampled_audio = np.zeros((1,1,768*level_mults[level])) + level
  print('new sample: ')
  print(sampled_audio, sampled_audio.shape)
  print('saving sample')

  if level_idx == len(levels)-1:
    if cur_sample == 0:
      print('cur_sample is zero, do not reach back')
      dd_sample = np.zeros((1,1,768*level_mults[level])) -1
    else:
      print("grabbing frames from last level 0 sample for current dd upsample")
      dd_sample = np.zeros((1,1,768*level_mults[level] + 1536)) -1
    print("sampling DD level")
    if cur_sample == 0:
      print('not doing xfade since cur_sample is 0')
    else:
      print("doing xfade, this involves creating fade in for current sample and reaching back and performing fade out on old sample, grab both audio snips and pass to function to perform fade and then insert")
    print('saving current level 0 sample as last_level_0, updating cur_sample')
    cur_sample += 768*level_mults[level]
    print(f"cur_sample: {cur_sample}")
    return cur_sample
  else:
    next_steps = level_mults[level]//level_mults[levels[level_idx+1]]
    print(f'next_steps: {next_steps}')
    for next_step in range(next_steps):
      cur_sample = sample_check(next_step, next_steps, level_idx+1, cur_sample=cur_sample, noise=sampled_audio, init=sampled_audio, levels=levels, level_mults=level_mults, context_windows=context_windows, final_audio=final_audio)
      print('updating context_window at current level for next sampling step, grabbing from finished audio')
    return cur_sample

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
