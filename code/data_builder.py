from utils import process_maestro_data
import torch
from torch.utils.data import Dataset

class MaestroDataset(Dataset):
    
    def __init__(self, datatype):
        self.processer = process_maestro_data(data_file="maestro-v3.0.0.zip")
        self.data = self.processer.read_csv_data()
        self.data = self.data[self.data.split == datatype].reset_index()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        midi_data = self.processer.read_midi_file(index)
        midi_data = midi_data[:28, :28]
        midi_data = torch.tensor(midi_data, dtype = torch.float)
        midi_data = torch.unsqueeze(midi_data, 0)
        return midi_data
        


