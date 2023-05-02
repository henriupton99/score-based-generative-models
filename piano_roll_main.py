import sys
sys.path.append("./code")
from utils import process_maestro_data
#processer = process_maestro_data(data_file = "maestro-v3.0.0.zip", start_pitch=56, fs = 1)
#res = processer.get_midi_metadata()
#print(res.shape)

#processer.plot_piano_roll(savefig=True)

from data_builder import MaestroDataset

dataset_train = MaestroDataset(
    datatype = "train",
    start_pitch=56,
    fs = 1)

print(dataset.__getitem__(0))

