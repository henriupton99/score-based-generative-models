import numpy as np
import pandas as pd
import zipfile
import pretty_midi
from zipfile import ZipFile

import matplotlib.pyplot as plt
import librosa

class process_maestro_data:
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.path = "./data/" + self.data_file
        
    def unzip_data(self):
        return ZipFile(self.path)

    def read_csv_data(self):
        zip_data = self.unzip_data()
        csv_path = self.data_file.replace(".zip", "") + \
            "/" + self.data_file.replace(".zip","") + \
            ".csv"
        with zip_data.open(csv_path) as file:
            csv_data = pd.read_csv(file)
        return csv_data
    
    def read_midi_file(self, index):
        zip_data = self.unzip_data()
        query = self.read_csv_data().loc[index]
        midi_filename = query.midi_filename
        midi_path = self.data_file.replace(".zip", "") + \
            "/" + midi_filename
        with zip_data.open(midi_path) as file:
            pm = pretty_midi.PrettyMIDI(file)
        return pm
        
    def plot_piano_roll(
        self,
        start_pitch,
        end_pitch,
        fs=100,
        savefig=False
    ):
        csv_data = self.read_csv_data()
        random_index = np.random.choice([k for k in range(len(csv_data))], size = 1)[0]
        pm = self.read_midi_file(random_index)
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                                hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                                fmin=pretty_midi.note_number_to_hz(start_pitch))
        plt.title("Piano Roll for " + csv_data.canonical_title.loc[random_index] + " (" + csv_data.canonical_composer.loc[random_index] + ")")
        if savefig is not False:
            plt.savefig("./figures/piano_roll_" + str(random_index) + ".png",
                        bbox_inches = "tight")