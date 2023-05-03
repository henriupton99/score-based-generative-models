import sys
sys.path.append("./code")
from utils import process_generated_samples
from config import config

samplers_list = ["Euler_Maruyama_sampler", "pc_sampler", "ode_sampler"]

for sampler_name in samplers_list:
    
    processer = process_generated_samples(
        data_file = "generated_samples/"+ sampler_name +"/generated_samples.pkl",
        sampler_name=sampler_name,
        start_pitch=config.start_pitch,
        fs = config.fs)

    processer.plot_piano_roll(savefig=True)



