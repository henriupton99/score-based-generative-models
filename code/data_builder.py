from utils import process_maestro_data
import torch
from torch.utils.data import Dataset

class MaestroDataset(Dataset):
    
    def __init__(self, datatype, start_pitch, fs):
        
        self.datatype = datatype
        self.start_pitch = start_pitch
        self.fs = fs
        
        self.processer = process_maestro_data(
            data_file="maestro-v3.0.0.zip",
            datatype = "train",
            start_pitch=self.start_pitch,
            fs=self.fs
            )
        self.data = self.processer.read_csv_data(datatype = self.datatype)
        
        self.midi_metadata = self.processer.get_midi_metadata()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        midi_data = self.midi_metadata[index, :, :]
        midi_data = torch.tensor(midi_data, dtype = torch.float)
        midi_data = torch.unsqueeze(midi_data, 0)
        return midi_data
        


