import numpy as np
import pandas as pd
from tqdm import tqdm
import zipfile
import pretty_midi
from zipfile import ZipFile

import matplotlib.pyplot as plt
import librosa

class process_maestro_data:
    
    def __init__(self, data_file, datatype, start_pitch, fs):
        
        self.datatype = datatype
        self.start_pitch = start_pitch
        self.end_pitch = self.start_pitch + 28
        self.fs = fs
        
        self.data_file = data_file
        self.path = "./data/" + self.data_file
        self.csv_data = self.read_csv_data(datatype = self.datatype)
        
    def unzip_data(self):
        return ZipFile(self.path)

    def read_csv_data(self, datatype):
        zip_data = self.unzip_data()
        csv_path = self.data_file.replace(".zip", "") + \
            "/" + self.data_file.replace(".zip","") + \
            ".csv"
        with zip_data.open(csv_path) as file:
            csv_data = pd.read_csv(file)
        
        csv_data = csv_data[csv_data.split == datatype].reset_index()
        
        return csv_data
    
    def read_midi_file(self, index):
        zip_data = self.unzip_data()
        query = self.csv_data.loc[index]
        midi_filename = query.midi_filename
        midi_path = self.data_file.replace(".zip", "") + \
            "/" + midi_filename
        with zip_data.open(midi_path) as file:
            pm = pretty_midi.PrettyMIDI(file)
            pm = pm.get_piano_roll(fs = self.fs)[self.start_pitch:self.end_pitch][:, :28]
        return pm
    
    def get_midi_metadata(self):
        midi_metadata = np.empty((len(self.csv_data), self.end_pitch-self.start_pitch, 28))
        for l in tqdm(range(len(midi_metadata))):
            midi_metadata[l, :, :] = self.read_midi_file(index = l)
        return midi_metadata
        
    def plot_piano_roll(
        self,
        savefig=False
    ):
        random_index = np.random.choice([k for k in range(len(self.csv_data))], size = 1)[0]
        pm = self.read_midi_file(random_index)
        plt.figure(figsize=(8, 4))
        pm = self.read_midi_file(random_index)
        librosa.display.specshow(pm,
                                hop_length=1, sr=self.fs, x_axis='time', y_axis='cqt_note',
                                fmin=pretty_midi.note_number_to_hz(self.start_pitch))
        plt.title("Piano Roll for " + self.csv_data.canonical_title.loc[random_index] + " (" + self.csv_data.canonical_composer.loc[random_index] + ")")
        if savefig is not False:
            plt.savefig("./figures/piano_roll_" + str(random_index) + ".png",
                        bbox_inches = "tight")