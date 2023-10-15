import torch
from jbdiff.utils import batch_preprocess
import pandas as pd

def create_data_reference_table(vqvae, dataloader, level, method='xq_mean'):
  vqvae.eval()
  assert not vqvae.training

  for batch in dataloader:
    x, fn = batch
    x = x.to('cuda')
    z_q, x_q = batch_preprocess(x, vqvae, level)

    if method == 'xq_mean':
      batch_means = torch.mean(torch.abs(x_q), dim=-1)
      return fn, batch_means
    else:
      raise Exception('Unknown method')
