#@title Training (double click to expand or collapse)

import sys
sys.path.append("./code")

from config import config
from data_builder import MaestroDataset

import torch
from functools import partial
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import ScoreNet, loss_fn
from sde import marginal_prob_std, diffusion_coeff
import functools
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

marginal_prob_std_fn = partial(marginal_prob_std, sigma=config.sigma)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=config.sigma)

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

print("LOADING TRAIN DATASET ...")
dataset_train = MaestroDataset(datatype = "train", start_pitch=config.start_pitch, fs=config.fs)
dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)

print("LOADING VALIDATION DATASET ...")
dataset_val = MaestroDataset(datatype = "validation", start_pitch=config.start_pitch, fs=config.fs)
dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=True)

optimizer = Adam(score_model.parameters(), lr=config.lr)
train_losses = []
val_losses = []

for epoch in range(config.n_epochs):
  print("EPOCH : " + str(epoch+1))
  avg_loss = 0.
  num_items = 0
  for x in tqdm(dataloader_train): 
    x = x.to(device)
    loss = loss_fn(score_model, x, marginal_prob_std_fn)
    optimizer.zero_grad()
    loss.backward()    
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  print('Average Loss Train: {:5f}'.format(avg_loss / num_items))
  train_losses.append(avg_loss / num_items)
  
  avg_loss = 0.
  num_items = 0
  for x in dataloader_val: 
    x = x.to(device)
    loss = loss_fn(score_model, x, marginal_prob_std_fn)
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  print('Average Loss Val: {:5f}'.format(avg_loss / num_items))
  val_losses.append(avg_loss / num_items)
  
  
  # Update the checkpoint after each epoch of training.
  torch.save(score_model.state_dict(), './data/ckpt.pth')

plt.figure(figsize=(10, 4))
x_axis = [k for k in range(1,config.n_epochs+1)]
plt.plot(x_axis, train_losses, color = "blue", label = "Train")
plt.plot(x_axis, val_losses, color = "orange", label = "Validation")
plt.xlabel("Epochs", size = 12)
plt.ylabel("Average Loss", size = 12)
plt.legend()
plt.savefig("./figures/training_history/train_val_losses.png", bbox_inches = "tight")
plt.show()