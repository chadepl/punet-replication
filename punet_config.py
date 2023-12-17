
BATCH_SIZE = 32
DEVICE = ["cpu", "mps", "cuda"][0]
EPOCHS = int(240000 * 32 * (1 / 8882))
NUM_CLASSES = 1
NUM_CHANNELS = [32, 64, 128, 256]  # original in paper   

LATENT_DIM=6  # # original in paper (3 for image2image VAE and 6 for PUnet)
NUM_CONVS_FCOMB=3 # original in paper   
BETA=1.0 # original in paper (10 for image2image VAE and 1 for PUnet)

VAL_FREQ = 10  # run validation every number of epochs
LEARNING_RATE = 1e-4
REG_WEIGHT = 1e-5

NUM_WORKERS = 32