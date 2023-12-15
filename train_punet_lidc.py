
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import matplotlib.pyplot as plt

from lidc_data import LIDCCrops

from train import train_punet
from punet import ProbabilisticUnet

##########
# Params #
##########

# For now shared to avoid problems in train/test scenarios

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

############
# Training #
############


transforms = dict(
    rand_elastic=dict(
        alpha=(0., 800.),  # as in the PUnet repo for cityscapes
        sigma=(25., 35.)  # as in the PUnet repo for cityscapes
        ),
    rand_affine=dict(
        degrees=(np.rad2deg(-np.pi/8), np.rad2deg(np.pi/8)),  # as in the PUnet repo for cityscapes
        translate=None,  # as in the PUnet repo for cityscapes (not used)
        scale_ranges=(0.8, 1.2),  # as in the PUnet repo for cityscapes
        shears=None  # as in the PUnet repo for cityscapes (not used)
        ),
    rand_crop=dict(
        output_size=(128, 128)   # as in the PUnet repo for cityscapes
        )
)
train_dataset = LIDCCrops(data_home="../data/lidc_crops", split="train", transform=transforms)
val_dataset = LIDCCrops(data_home="../data/lidc_crops", split="val", transform=None)

train_punet(train_dataset=train_dataset,
            batch_size_train=BATCH_SIZE,
            val_dataset=val_dataset,
            batch_size_val=BATCH_SIZE,
            epochs=EPOCHS,
            num_classes=NUM_CLASSES,
            num_channels_unet=NUM_CHANNELS,
            latent_dim=LATENT_DIM,
            no_convs_fcomb=NUM_CONVS_FCOMB,
            beta=BETA,
            train_id=f"NicolasTest-lidc-patches-lv3",
            device=DEVICE, num_workers=NUM_WORKERS)



