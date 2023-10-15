import torch
from jbdiff.utils import batch_preprocess
import pandas as pd
import numpy as np
from tqdm import tqdm


def create_data_reference_table(vqvae, dataloader, level, method='zq_count'):
  vqvae.eval()
  assert not vqvae.training
  fns = list()
  final = list()

  for batch in tqdm(dataloader):
    x, fn = batch
    x = x.to('cuda')
    z_q, x_q = batch_preprocess(x, vqvae, level)

    if method == 'xq_mean':
      batch_means = torch.mean(torch.abs(x_q), dim=-1)
      fns += list(fn)
      final.append(batch_means.to('cpu').numpy())
    elif method == 'zq_count':
      b = z_q.shape[0]
      z_q = z_q.to('cpu').numpy()
      batch_counts = np.zeros((b, 2048))
      for i, z in enumerate(z_q):
        unique, counts = np.unique(z, return_counts=True)
        batch_counts[i,unique] = counts
      fns += list(fn)
      final.append(batch_counts)
    elif method == 'both':
      final = dict(xq_mean=list(), zq_count=list())
      fns += list(fn)
      # xq_mean
      batch_means = torch.mean(torch.abs(x_q), dim=-1)
      final['xq_mean'].append(batch_means.to('cpu').numpy())
      # zq_count
      b = z_q.shape[0]
      z_q = z_q.to('cpu').numpy()
      batch_counts = np.zeros((b, 2048))
      for i, z in enumerate(z_q):
        unique, counts = np.unique(z, return_counts=True)
        batch_counts[i,unique] = counts
      final['zq_count'].append(batch_counts)
    else:
      raise Exception('Unknown method')

  return final, fns


def create_sample_reference(vqvae, audio, level, method='zq_count'):
  vqvae.eval()
  assert not vqvae.training

  x = torch.tensor(x, device='cuda')
  x = x.unsqueeze(0)
  z_q, x_q = batch_preprocess(x, vqvae, level)

  if method == 'xq_mean':
    sample_mean = torch.mean(torch.abs(x_q), dim=-1)
    sample_mean = sample_mean.to('cpu').numpy()
    final = sample_mean
  elif method == 'zq_count':
    b = z_q.shape[0]
    z_q = z_q.to('cpu').numpy()
    batch_counts = np.zeros((b, 2048))
    unique, counts = np.unique(z_q, return_counts=True)
    batch_counts[:,unique] = counts
    final = batch_counts
  else:
    raise Exception('Unknown method')

  return final


def create_latent_reference(z_q, x_q, method='zq_count'):
  if method == 'xq_mean':
    sample_mean = torch.mean(torch.abs(x_q), dim=-1)
    sample_mean = sample_mean.to('cpu').numpy()
    final = sample_mean
  elif method == 'zq_count':
    b = z_q.shape[0]
    z_q = z_q.to('cpu').numpy()
    batch_counts = np.zeros((b, 2048))
    unique, counts = np.unique(z_q, return_counts=True)
    batch_counts[:,unique] = counts
    final = batch_counts
  else:
    raise Exception('Unknown method')

  return final


def get_top_n_similarities(sample_ref, data_ref, fns, loss_fn, top_n=100):
  assert len(data_ref) == len(fns), "data_ref and fns not same size"
  losses = list()
  for data in tqdm(data_ref):
    losses.append(loss_fn(torch.tensor(sample_ref), torch.tensor(data)).to('cpu').numpy())
  losses = np.array(losses)
  final = np.argsort(losses)[:top_n]
  fns_list = list()
  for f in final:
    fns_list.append(fns[f])
  return final, fns_list
