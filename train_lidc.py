
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

run_id = f"NicolasTest-lidc-patches-lv3"

from punet_config import DEVICE, NUM_WORKERS
from punet_config import NUM_CLASSES, NUM_CHANNELS, LATENT_DIM, NUM_CONVS_FCOMB
from punet_config import BATCH_SIZE, EPOCHS, BETA, VAL_FREQ, LEARNING_RATE, REG_WEIGHT

########
# Data #
########

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

############
# Training #
############

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
            learning_rate=LEARNING_RATE, 
            val_freq=VAL_FREQ,
            reg_weight=REG_WEIGHT,
            train_id=run_id,
            device=DEVICE, num_workers=NUM_WORKERS)



