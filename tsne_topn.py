import os
import argparse
from jbdiff.utils import read_yaml_file, parse_diff_conf, make_jb, ExceptionCallback, DemoCallback, JBDiffusion
from jbdiff.attribution_utils import create_data_reference_table, create_sample_reference, create_latent_reference, get_top_n_similarities
import torch as t
import pytorch_lightning as pl
from sklearn.manifold import TSNE
from torch.nn import L1Loss, MSELoss
from einops import rearrange
import numpy as np
import pandas as pd

CONFIG_FILE = 'jbdiff-v1.yaml'
conf = read_yaml_file(CONFIG_FILE)

# Load VQVAE args from conf
vqvae_conf = conf['model']['vqvae']
context_mult = vqvae_conf['context_mult']
batch_size = vqvae_conf['batch_size']
aug_shift = vqvae_conf['aug_shift']
base_tokens = vqvae_conf['base_tokens']

num_workers = 8
level = 2
audio_dir = '/home/ubuntu/wavs'
batch_size = 512

vqvae, dataloader, hps = make_jb(audio_dir, level, batch_size, base_tokens, context_mult, aug_shift, num_workers)

times = 3
method = 'both'
meths = ['xq_mean', 'zq_count']
data_refs = dict(xq_mean=list(), zq_count=list())
fn_dict = dict(xq_mean=list(), zq_count=list())
loss_dict = dict()
for i in range(times):
  data_ref, fns = create_data_reference_table(vqvae, dataloader, level, method=method)
  for meth in meths:
    meth_ref = np.array(data_ref[meth])
    meth_ref = rearrange(meth_ref, "n b s -> (n b) s")
    data_refs[meth].append(meth_ref)
    fn_dict[meth] += fns

for meth in meths:
  data_refs[meth] = np.array(data_refs[meth])
  data_refs[meth] = rearrange(data_refs[meth], "n b s -> (n b) s")

loss_dict['xq_mean'] = MSELoss()
loss_dict['zq_count'] = L1Loss()

b = data_refs['xq_mean'].shape[0]
rand_idx = np.random.choice(b)
print(f"Random Index: {rand_idx}\nFilename: {fn_dict['xq_mean'][rand_idx]} {fn_dict['zq_count'][rand_idx]}")

series_list = list()
for method in methods:
  sample = data_refs[method][rand_idx]
  ids, fn_l = get_top_n_similarities(sample, data_refs[method], fn_dict[method], loss_dict[method], top_n=200)
  series_list.append(pd.Series(fn_l).value_counts().sort_values())

full_series = pd.concat(series_list, axis=1).fillna(0).sum(1).sort_values()

full_series[-10:]
