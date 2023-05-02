#@title Sampling (double click to expand or collapse)

import sys
import torch
import functools
import pandas as pd
sys.path.append("./code")
from config import config
from model import ScoreNet
from sampler import Euler_Maruyama_sampler
from sde import marginal_prob_std, diffusion_coeff

## Load the pre-trained checkpoint from disk.
device = 'cpu' #@param ['cuda', 'cpu'] {'type':'string'}
sys.path.append("./data")
ckpt = torch.load('./data/ckpt.pth', map_location=device)

marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=config.sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=config.sigma)

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)
score_model.load_state_dict(ckpt)

sample_batch_size = 12
sampler = Euler_Maruyama_sampler

## Generate samples using the specified sampler : 
samples = sampler(score_model, 
                  marginal_prob_std_fn,
                  diffusion_coeff_fn, 
                  sample_batch_size, 
                  device=device)

samples = torch.where(samples > 0, 1, 0)
piano_roll = [samples[l, 0, :, :] for l in range(sample_batch_size)]
df_res = pd.DataFrame(data = {"piano_roll" : piano_roll})
df_res.to_csv('./data/generated_samples.csv')