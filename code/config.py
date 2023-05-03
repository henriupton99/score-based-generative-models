class Config:
    
    def __init__(self):
        ## Preprocessing hyperparameters :
        self.start_pitch = 56
        self.fs = 1
        
        ## Training hyperparameters :
        self.batch_size = 32
        self.lr = 1e-4
        self.n_epochs = 50
        
        ## Noising/Sampling hyperparameters :
        self.num_steps = 500
        self.sigma = 25.0
        self.signal_to_noise_ratio = 0.16
        self.error_tolerance = 1e-5
        self.activation_threshold = 10.0

config = Config()