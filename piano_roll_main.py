import sys
sys.path.append("./code")
from utils import process_generated_samples
from config import config

processer = process_generated_samples(
    data_file = "generated_samples.pkl",
    start_pitch=config.start_pitch,
    fs = config.fs)

processer.plot_piano_roll(savefig=True)



