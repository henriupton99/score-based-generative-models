class Config:
    
    def __init__(self):
        ## Preprocessing hyperparameters :
        self.start_pitch = 56
        self.fs = 1
        
        ## Training hyperparameters :
        self.batch_size = 32
        self.lr = 1e-4
        self.n_epochs = 50
        
        ## noising hyperparameters :
        self.sigma = 25

config = Config()