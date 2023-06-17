import os
import argparse
from jbdiff.utils import read_yaml_file, parse_diff_conf, make_jb, ExceptionCallback, DemoCallback, JBDiffusion
import torch as t
import pytorch_lightning as pl


#----------------------------------------------------------------------------

# Change config file to change hyperparams
CONFIG_FILE = 'jbdiff-v1.yaml'

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
  level = kwargs['jb_level']
  audio_dir = kwargs['train_data']
  save_path = kwargs['ckpt_save_location']
  log_to_wandb = kwargs['log_to_wandb']
  demo_every = kwargs['demo_every']
  num_demos = kwargs['num_demos']
  demo_seconds = kwargs['demo_seconds']
  demo_steps = kwargs['demo_steps']
  embedding_weight = kwargs['embedding_weight']
  resume_pkl = kwargs['resume_network_pkl']
  num_workers = kwargs['num_workers']
  ckpt_every = kwargs['ckpt_every']

  # Load diffusion config
  diffusion_conf = conf['model']['diffusion'][level]
  diffusion_conf = parse_diff_conf(diffusion_conf)
  diffusion_conf['embedding_max_length'] = context_mult*base_tokens

  # Load vqvae, dataloader, and their hyperparams
  vqvae, dataloader, hps = make_jb(audio_dir, level, batch_size, base_tokens, context_mult, aug_shift, num_workers)
  print('sample_length: ', hps.sample_length)
  # Load and Train Diffusion Model
  sr = hps.sr
  demo_samples = demo_seconds*sr
  project_name = f'jbdiff_level_{level}'

  # Args for demos while training
  class DemoArgs:
    def __init__(self): 
      self.demo_every = demo_every
      self.num_demos = num_demos
      self.sample_size = demo_samples
      self.demo_steps = demo_steps
      self.sample_rate = sr
      self.base_samples = hps.sample_length
      self.base_tokens = base_tokens
      self.dirpath = save_path
      self.embedding_scale = embedding_weight

  demo_args = DemoArgs()

  # Defining callbacks and loggers
  exc_callback = ExceptionCallback()
  ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=ckpt_every, save_top_k=-1, dirpath=save_path)
  demo_callback = DemoCallback(demo_args)

  # Instantiate model
  diffusion_model = JBDiffusion(vqvae=vqvae, level=level, diffusion_kwargs=diffusion_conf)
  print(diffusion_model.diffusion)

  # Call trainer with WandB logging
  if log_to_wandb:
    wandb_logger = pl.loggers.WandbLogger(project=project_name, log_model='all')
    wandb_logger.watch(diffusion_model)
    diffusion_trainer = pl.Trainer(
      devices=1,
      accelerator="gpu",
      precision=16,
      accumulate_grad_batches=4, 
      callbacks=[ckpt_callback, demo_callback, exc_callback],
      logger=wandb_logger,
      log_every_n_steps=1,
      max_epochs=10000000,
    )

  # Call trainer without WandB logging
  else:
    diffusion_trainer = pl.Trainer(
      devices=1,
      accelerator="gpu",
      precision=16,
      accumulate_grad_batches=4, 
      callbacks=[ckpt_callback, demo_callback, exc_callback],
      log_every_n_steps=1,
      max_epochs=10000000,
    )

  # Train from scratch
  if resume_pkl is None:
    diffusion_trainer.fit(diffusion_model, dataloader)
  # Train from checkpoint
  else:
    diffusion_trainer.fit(diffusion_model, dataloader, ckpt_path=resume_pkl)

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

  # Train deepest level JBDiff on personal music library
  python train.py --train-data ./wavs --jb-level 2 --ckpt-save-location ./ckpts 

  # Resume training middle layer of JBDiff from checkpoint
  python train.py --train-data ./wavs --jb-level 1 --ckpt-save-location ./ckpts --resume-network-pkl ./ckpts/ckpt1.ckpt

'''


def main():
  parser = argparse.ArgumentParser(
    description = 'Train JB Latent Diffusion Model on custom dataset', 
    epilog=_examples, 
    formatter_class=argparse.RawDescriptionHelpFormatter
    )
  parser.add_argument('--train-data', help='Location of training data, MAKE SURE all files are .wav format and the same sample rate', required=True, metavar='DIR', type=_path_exists)
  parser.add_argument('--jb-level', help='Which level of Jukebox VQ-VAE to train on (start with 2 and work back to 0)', required=True, type=int)
  parser.add_argument('--ckpt-save-location', help='Location to save network checkpoints', required=True, metavar='FILE', type=_path_exists)
  parser.add_argument('--log-to-wandb', help='T/F whether to log to weights and biases', default=True, metavar='BOOL', type=_str_to_bool)
  parser.add_argument('--resume-network-pkl', help='Location of network pkl to resume training from', default=None, metavar='FILE', type=_path_exists)
  parser.add_argument('--num-workers', help='Number of workers dataloader should use, depends on machine, if you get a message about workers being a bottleneck, adjust to recommended size here', default=12, type=int)
  parser.add_argument('--demo-every', help='Number of training steps per demo', default=2500, type=int)
  parser.add_argument('--num-demos', help='Batch size of demos, must be <= batch_size of training', default=4, type=int)
  parser.add_argument('--demo-seconds', help='Length of each demo in seconds', default=10, type=int)
  parser.add_argument('--demo-steps', help='Number of diffusion steps in demo', default=250, type=int)
  parser.add_argument('--embedding-weight', help='Conditioning embedding weight between 0-1 for demos', default=0.36, type=float)
  parser.add_argument('--ckpt-every', help='Number of training steps per checkpoint', default=5000, type=int)
  args = parser.parse_args()

  run(**vars(args))


#----------------------------------------------------------------------------


if __name__ == "__main__":
    main()


#----------------------------------------------------------------------------
