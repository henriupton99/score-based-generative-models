import sys
sys.path.append("./code")
from utils import process_maestro_data
processer = process_maestro_data(data_file = "maestro-v3.0.0.zip")
processer.plot_piano_roll(start_pitch=56, end_pitch=70, savefig=True)
print("done")