#@title Training (double click to expand or collapse)

import sys
sys.path.append("./code")

from data_builder import MaestroDataset

import torch
from functools import partial
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from model import ScoreNet, loss_fn
from sde import marginal_prob_std, diffusion_coeff
import functools
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sigma =  25.0
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma)

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

n_epochs = 5
batch_size =  32 
lr=1e-4

dataset = MaestroDataset(datatype = "train")
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = Adam(score_model.parameters(), lr=lr)
for epoch in range(n_epochs):
  avg_loss = 0.
  num_items = 0
  for x in data_loader: 
    x = x.to(device)
    loss = loss_fn(score_model, x, marginal_prob_std_fn)
    optimizer.zero_grad()
    loss.backward()    
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  # Print the averaged training loss so far.
  tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
  # Update the checkpoint after each epoch of training.
  torch.save(score_model.state_dict(), './data/ckpt.pth')