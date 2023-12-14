
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

if False:  # we want to train a network or not

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


#######################
# Qualitative Testing #
#######################

if True:

    rng = np.random.default_rng(seed=42)

    NUM_IMGS = 5
    NUM_SAMPLES = 5
    PROB_ISOVAL = 0.8

    # Load model
    checkpoint_path = Path("training_logs/NicolasWork-lidc-patches-lv3/model_checkpoints//checkpoint_epoch-40.pth")
    state_dict = torch.load(str(checkpoint_path), map_location=torch.device(DEVICE))

    net = ProbabilisticUnet(num_input_channels=1,
                            num_classes=NUM_CLASSES,
                            num_filters=NUM_CHANNELS,
                            latent_dim=LATENT_DIM,
                            no_convs_fcomb=NUM_CONVS_FCOMB,
                            beta=BETA,
                            device=DEVICE)
    net.to(DEVICE)

    net.load_state_dict(state_dict=state_dict)

    # Get random images
    test_dataset = LIDCCrops(data_home="../data/lidc_crops", split="test", transform=dict(resize=dict(output_size=(128, 128))))
    metadatas, imgs, segs = zip(*[test_dataset[i] for i in rng.choice(np.arange(len(test_dataset)), NUM_IMGS, replace=False)])
    
    imgs = [img.unsqueeze(dim=0) for img in imgs]    
    imgs = torch.cat(imgs, dim=0)

    segs = [seg.unsqueeze(dim=0).unsqueeze(dim=0) for seg in segs]    
    segs = torch.cat(segs, dim=0)

    imgs = imgs.to(DEVICE)

    probs = []
    preds = []
    for nsample in range(NUM_SAMPLES):
        net(imgs, None, training=False)  # Run net (this initializes the unet features and the latent space)
        prob = net.sample(testing=True)  # samples a segmentation using the unet features + the latent space
        
        pred = prob > PROB_ISOVAL

        # Use when num classes > 1
        # prob = nn.Softmax(dim=1)(sample)
        # pred = torch.argmax(probs, dim=1)

        probs.append(prob)
        preds.append(pred)


    # plot
    map_to_vis = [probs, preds][0]

    fig, axs = plt.subplots(nrows=NUM_IMGS, ncols=NUM_SAMPLES+2, layout="tight")

    # - plot imgs
    for img_i in range(NUM_IMGS):
        axs[img_i, 0].imshow(imgs.cpu().numpy()[img_i, 0], cmap="gray")
        axs[img_i, 0].set_ylabel(f"Image {img_i}")

    # - plot gt
    axs[0, 1].set_title("GT 1/4")
    for img_i in range(NUM_IMGS):
        axs[img_i, 1].imshow(segs.cpu().numpy()[img_i, 0], cmap="gray")        

    # - plot samples
    for sample_i in range(NUM_SAMPLES):
        axs[0, sample_i + 2].set_title(f"Sample {sample_i}")
        for img_i in range(NUM_IMGS):
            axs[img_i, sample_i + 2].imshow(map_to_vis[sample_i].detach().cpu().numpy()[img_i, 0])

    for ax in axs.flatten():
        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
