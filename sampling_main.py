#@title Sampling (double click to expand or collapse)

import sys
import torch
import functools
import pandas as pd
sys.path.append("./code")
from config import config
from model import ScoreNet
from sampler import Euler_Maruyama_sampler, pc_sampler, ode_sampler
from sde import marginal_prob_std, diffusion_coeff

## Load the pre-trained checkpoint from disk.
device = 'cpu' #@param ['cuda', 'cpu'] {'type':'string'}
sys.path.append("./data")
ckpt = torch.load('./data/model.pth', map_location=device)

marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=config.sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=config.sigma)

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)
score_model.load_state_dict(ckpt)

sample_batch_size = 12
samplers_list = [Euler_Maruyama_sampler, pc_sampler, ode_sampler]
for sampler in samplers_list:
    print(str(sampler.__name__))

for sampler in samplers_list:
    sampler_name = str(sampler.__name__)
    print("SAMPLING FOR " + sampler_name.upper() + " :")
    ## Generate samples using the specified sampler : 
    
    if sampler_name == "Euler_Maruyama_sampler":
        samples = sampler(score_model, 
                        marginal_prob_std_fn,
                        diffusion_coeff_fn, 
                        num_steps=config.num_steps, 
                        batch_size=sample_batch_size, 
                        device=device)
        
    if sampler_name == "pc_sampler":
        samples = pc_sampler(score_model, 
                        marginal_prob_std_fn,
                        diffusion_coeff_fn,
                        snr=config.signal_to_noise_ratio, 
                        num_steps=config.num_steps,
                        batch_size=sample_batch_size,                 
                        device=device,
                        eps=1e-3)
    
    if sampler_name == "ode_sampler":
        samples = ode_sampler(score_model,
                        marginal_prob_std_fn,
                        diffusion_coeff_fn,
                        atol=config.error_tolerance, 
                        rtol=config.error_tolerance, 
                        batch_size=sample_batch_size, 
                        device=device, 
                        z=None,
                        eps=1e-3)

    samples = torch.where(samples > config.activation_threshold, 1, 0)
    piano_roll = [samples[l, 0, :, :] for l in range(sample_batch_size)]
    df_res = pd.DataFrame(data = {"piano_roll" : piano_roll})
    df_res.to_pickle('./data/generated_samples/' + sampler_name + "/generated_samples.pkl")