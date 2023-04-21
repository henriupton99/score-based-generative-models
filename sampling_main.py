#@title Sampling (double click to expand or collapse)

import sys
import torch
import functools
sys.path.append("./code")
from model import ScoreNet
from sampler import Euler_Maruyama_sampler
from sde import marginal_prob_std, diffusion_coeff

## Load the pre-trained checkpoint from disk.
device = 'cpu' #@param ['cuda', 'cpu'] {'type':'string'}
sys.path.append("./data")
ckpt = torch.load('./data/ckpt.pth', map_location=device)

sigma=25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)
score_model.load_state_dict(ckpt)

sample_batch_size = 64 #@param {'type':'integer'}
sampler = Euler_Maruyama_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

## Generate samples using the specified sampler.
samples = sampler(score_model, 
                  marginal_prob_std_fn,
                  diffusion_coeff_fn, 
                  sample_batch_size, 
                  device=device)

print(samples.shape)
samples.to_csv('./data/samples_test.csv')